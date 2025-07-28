from numpy import ndarray
from src.skwiz.box_classifier.gru import GRUBoxClassifier
from src.skwiz.box_classifier.lstm import LSTMBoxClassifier
from src.skwiz.layoutlm.default_layoutlm_config import DEFAULT_LAYOUTLM_CONFIG
import torch
import os
import json
import pickle
import safetensors.torch
from PIL import Image
from typing import Any, Dict, List, Literal
from transformers import AutoProcessor  # type: ignore[import-untyped]

from src.skwiz.layoutlm.layoutlmv3_and_features_classification import (
    LayoutLMv3AndFeaturesClassification,
)
from src.skwiz.constants import AUTO_PROCESSOR_LOCAL_PATH

from src.skwiz.utils import (
    box_classifier_post_processing,
    get_box_classifier_inference_batches,
    get_box_classifier_inference_predictions,
    get_page_preprocess_input_splited_by_windows,
    get_layoutlm_inference_batches,
    get_layoutlm_inference_predictions,
    post_processing_classification,
    normalise_ocr_bounding_box,
    post_processing_vectors,
    preprocess_single_task_box_classifier_for_inference,
)
from src.types.inference import (
    InferenceBoundingBox,
    InferenceOutput,
    InferencePageClassification,
)
from src.types.ocr import Ocr, PageInfo
from src.types.training import TrainingConfig, TrainingType
from src.types.layoutlm import ProcessedLayoutLMInput

MODEL_FOLDER = "/var/www/models/skwiz"

MAX_PARALLEL_COMPUTATION = 8
CONFIG_FILE = "config.json"
EXTRACTION_LABEL_IDS_FILE = "extraction_label_ids.pickle"
CLASSIFICATION_LABEL_IDS_FILE = "classification_label_ids.pickle"
LAYOUT_LM_MODEL_FILE = "model.safetensors"
BOX_CLASSIFIER_MODEL_FILE = "box_classifier_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_name: str):
    print(f"Loading model: {model_name}", flush=True)
    model_path = os.path.join(MODEL_FOLDER, model_name)

    # Config
    config_path = os.path.join(model_path, CONFIG_FILE)
    with open(config_path, "r", encoding="utf-8") as json_file:
        training_config: TrainingConfig = json.loads(json_file.read())

    # Label IDs
    extraction_label_ids_path = os.path.join(model_path, EXTRACTION_LABEL_IDS_FILE)
    with open(extraction_label_ids_path, "rb") as f:
        extraction_label_ids = pickle.load(f)
        extraction_id2label = extraction_label_ids["id2label"]
        extraction_label2id = extraction_label_ids["label2id"]
    classification_label_ids_path = os.path.join(
        model_path, CLASSIFICATION_LABEL_IDS_FILE
    )
    with open(classification_label_ids_path, "rb") as f:
        classification_label_ids = pickle.load(f)
        classification_id2label = classification_label_ids["id2label"]
        classification_label2id = classification_label_ids["label2id"]

    # Processor
    processor = AutoProcessor.from_pretrained(
        AUTO_PROCESSOR_LOCAL_PATH, apply_ocr=False, local_files_only=True
    )
    training_config_layoutlm = training_config.get("layoutlm", {})
    training_config_layoutlm_architecture_config = training_config_layoutlm.get(
        "architectureConfig", {}
    )
    training_config_max_position_embeddings = training_config_layoutlm_architecture_config.get(
        "max_position_embeddings", None
    )

    if training_config_max_position_embeddings is not None:
        processor.tokenizer.model_max_length = training_config_max_position_embeddings - 2

    # Layoutlm model
    layoutlm_model = LayoutLMv3AndFeaturesClassification(
        config=training_config,
        extraction_label2id=extraction_label2id,
        classification_label2id=classification_label2id,
    )
    model_tensors_path = os.path.join(model_path, LAYOUT_LM_MODEL_FILE)
    model_weights = safetensors.torch.load_file(model_tensors_path)
    layoutlm_model.load_state_dict(model_weights)
    layoutlm_model.to(device)

    # Box classifier model
    should_have_box_classifier = (
        training_config.get("type") == TrainingType.EXTRACTION.value
        and len(extraction_label2id.keys()) > 0
    )
    if should_have_box_classifier:
        box_classifier_config = training_config.get(
            'boxClassifier', {})
        box_classifier_model_config = box_classifier_config.get(
            'model', {})
        model_type = box_classifier_model_config.get('type', 'gru')
        input_size = training_config_layoutlm.get(
            "architectureConfig", {}
        ).get('hidden_size', DEFAULT_LAYOUTLM_CONFIG.get('hidden_size', 768)) or 768
        hidden_size = box_classifier_model_config.get('hidden_size', 384)
        num_layers = box_classifier_model_config.get('num_layers', 1)
        dropout = box_classifier_model_config.get('dropout', 0)
        bidirectional = box_classifier_model_config.get('bidirectional', True)

        ocr_tags = training_config.get('tagging', {}).get('ocrTags', [])
        num_tags = len(ocr_tags)
        box_base_model = GRUBoxClassifier if model_type == 'gru' else LSTMBoxClassifier

        box_classifier_model: torch.nn.Module = box_base_model(
            input_size=input_size,
            extraction_label2id=extraction_label2id,
            hidden_size=hidden_size,
            n_tags=num_tags,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        box_model_tensors_path = os.path.join(model_path, BOX_CLASSIFIER_MODEL_FILE)
        box_model_weights = torch.load(box_model_tensors_path, map_location=device)
        box_classifier_model.load_state_dict(box_model_weights)
        box_classifier_model.to(device)

    return (
        box_classifier_model if should_have_box_classifier else None,
        layoutlm_model,
        processor,
        training_config,
        extraction_id2label,
        classification_id2label,
    )


def predict_with_model(
    layoutlm_model,
    processor,
    training_config,
    extraction_id2label,
    classification_id2label,
    pil_images: List[Image.Image],
    ocr_results: Ocr,
    box_classifier_model = None
) -> InferenceOutput:
    # Inference
    document_preprocess_input_windows: List[ProcessedLayoutLMInput] = []
    document_preprocess_input_windows_page_indexes: List[int] = []
    for page_index, page_pil_image in enumerate(pil_images):
        page_ocr = ocr_results["pages"][page_index]
        page_bbs_text = [bb["text"] for bb in page_ocr["boundingBoxes"]]
        page_bbs_coordinates = [
            normalise_ocr_bounding_box(bb["coordinates"])
            for bb in page_ocr["boundingBoxes"]
        ]
        page_bbs_tags: List[List[int]] = [[] for _ in page_ocr["boundingBoxes"]]

        page_preprocess_input_window = get_page_preprocess_input_splited_by_windows(
            page_pil_image,
            page_bbs_text,
            page_bbs_coordinates,
            page_bbs_tags,
            processor,
            training_config,
        )
        document_preprocess_input_windows = (
            document_preprocess_input_windows + page_preprocess_input_window
        )
        document_preprocess_input_windows_page_indexes = (
            document_preprocess_input_windows_page_indexes
            + [page_index] * len(page_preprocess_input_window)
        )

    # Inference batches
    inference_batches = get_layoutlm_inference_batches(
        preprocess_input_windows=document_preprocess_input_windows,
        batch_size=MAX_PARALLEL_COMPUTATION,
        device=device,
    )

    _, outputs_classifier, output_layoutlm_vectors = get_layoutlm_inference_predictions(
        model=layoutlm_model, batches=inference_batches
    )

    page_indexes = [i for i in range(len(pil_images))]
    ocr_pages = ocr_results["pages"]

    classification_prediction_pages = post_processing_classification(
        outputs_classifier=outputs_classifier,
        classification_id2label=classification_id2label,
        page_indexes=page_indexes,
        document_preprocess_input_windows_page_indexes=document_preprocess_input_windows_page_indexes,
    )

    box_classifier_output: List[InferenceBoundingBox] = []
    if box_classifier_model is not None:
        box_classifier_input_size = box_classifier_model.input_size
        layoutlm_vector_by_bbox = post_processing_vectors(
            vectors_output=output_layoutlm_vectors,
            ocr_pages=ocr_pages,
            page_indexes=page_indexes,
            window_page_indexes=document_preprocess_input_windows_page_indexes,
            document_preprocess_input_windows=document_preprocess_input_windows,
        )
        preprocess_box_classifier_pages, window_page_indexes = preprocess_single_task_box_classifier_for_inference(
            training_config=training_config,
            input_size=box_classifier_input_size,
            layoutlm_vectors_by_bbox=layoutlm_vector_by_bbox,
            ocr_pages=ocr_pages,
            page_indexes=page_indexes
        )

        box_classifier_inference_batches = get_box_classifier_inference_batches(
            preprocess_box_classifier_windows=preprocess_box_classifier_pages,
            batch_size=MAX_PARALLEL_COMPUTATION,
            device=device
        )

        box_classifier_predictions = get_box_classifier_inference_predictions(
            inference_model=box_classifier_model,
            batches=box_classifier_inference_batches
        )

        box_classifier_output = box_classifier_post_processing(
            box_classifier_predictions=box_classifier_predictions,
            extraction_id2label=extraction_id2label,
            page_indexes=page_indexes,
            ocr_pages=ocr_pages,
            window_page_indexes=window_page_indexes,
            preprocessed_box_classifier_windows=preprocess_box_classifier_pages
        )

    return {
        "boundingBoxes": box_classifier_output,
        "classification": classification_prediction_pages,
    }


ClassifierModelType = Literal[
    "classifier-lng",
    "classifier-oqgn",
]
ExtractorModelType = Literal[
    "extractor-oqgn",
    "extractor-oqgn-tables-del",
]
classifier_models: Dict[ClassifierModelType, Any] = {
    "classifier-lng": get_model("classifier-lng"),
    "classifier-oqgn": get_model("classifier-oqgn"),
}

extractor_models: Dict[ExtractorModelType, Any] = {
    "extractor-oqgn": get_model("extractor-oqgn"),
    "extractor-oqgn-tables-del": get_model("extractor-oqgn-tables-del"),
}



def classify_page(
    model_name: ClassifierModelType, image: ndarray, page_ocr: PageInfo
) -> InferencePageClassification:
    (
        _box_classifier_model,
        layoutlm_model,
        processor,
        training_config,
        extraction_id2label,
        classification_id2label,
    ) = classifier_models[model_name]

    results = predict_with_model(
        layoutlm_model,
        processor,
        training_config,
        extraction_id2label,
        classification_id2label,
        [Image.fromarray(image)],
        {"pages": [page_ocr]},
    )

    return results["classification"][0]

    # return [process_classification_page(classification) for classification in classif]


def extract_page(
    model_name: ExtractorModelType,
    image: ndarray,
    page_ocr: PageInfo,
) -> List[InferenceBoundingBox]:
    (
        box_classifier_model,
        layoutlm_model,
        processor,
        training_config,
        extraction_id2label,
        classification_id2label,
    ) = extractor_models[model_name]

    results = predict_with_model(
        layoutlm_model,
        processor,
        training_config,
        extraction_id2label,
        classification_id2label,
        [Image.fromarray(image)],
        {"pages": [page_ocr]},
        box_classifier_model,
    )

    return results["boundingBoxes"]

    # return process_extraction_page(
    #     results["boundingBoxes"],
    #     page_ocr,
    #     fields,
    #     mergeable_fields,
    #     tables,
    # )
