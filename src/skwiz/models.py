from numpy import ndarray
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
    get_page_preprocess_input_splited_by_windows,
    get_layoutlm_inference_batches,
    get_layoutlm_inference_predictions,
    post_processing_extraction,
    post_processing_classification,
    normalise_ocr_bounding_box,
)
from src.types.inference import (
    InferenceBoundingBox,
    InferenceOutput,
    InferencePageClassification,
)
from src.types.ocr import Ocr, PageInfo
from src.types.training import TrainingConfig
from src.types.layoutlm import ProcessedLayoutLMInput

MODEL_FOLDER = "/var/www/models/skwiz"

MAX_PARALLEL_COMPUTATION = 8
CONFIG_FILE = "config.json"
EXTRACTION_LABEL_IDS_FILE = "extraction_label_ids.pickle"
CLASSIFICATION_LABEL_IDS_FILE = "classification_label_ids.pickle"
LAYOUT_LM_MODEL_FILE = "model.safetensors"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_name: str):
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
    training_config_max_position_embeddings = training_config.get("layoutlmv3", {}).get(
        "max_position_embeddings", None
    )
    if training_config_max_position_embeddings is not None:
        processor.tokenizer.model_max_length = (
            training_config_max_position_embeddings - 2
        )

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

    return (
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

    output_extraction, outputs_classifier = get_layoutlm_inference_predictions(
        model=layoutlm_model, batches=inference_batches
    )

    bounding_boxes_with_predictions = post_processing_extraction(
        output_extraction=output_extraction,
        extraction_id2label=extraction_id2label,
        ocr_pages=ocr_results["pages"],
        window_page_indexes=document_preprocess_input_windows_page_indexes,
        document_preprocess_input_windows=document_preprocess_input_windows,
    )

    classification_prediction_pages = post_processing_classification(
        outputs_classifier=outputs_classifier,
        classification_id2label=classification_id2label,
        page_indexes=[i for i in range(len(pil_images))],
        document_preprocess_input_windows_page_indexes=document_preprocess_input_windows_page_indexes,
    )

    return {
        "boundingBoxes": bounding_boxes_with_predictions,
        "classification": classification_prediction_pages,
    }


ClassifierModelType = Literal[
    "classifier-lng",
    "classifier-oqgn",
]
ExtractorModelType = Literal[
    "extractor-oqgn",
    "extractor-oqgn-tables-del",
    "extractor-oqgn-tables-ooc",
]
classifier_models: Dict[ClassifierModelType, Any] = {
    "classifier-lng": get_model("classifier-lng"),
    "classifier-oqgn": get_model("classifier-oqgn"),
}

extractor_models: Dict[ExtractorModelType, Any] = {
    "extractor-oqgn": get_model("extractor-oqgn"),
    "extractor-oqgn-tables-del": get_model("extractor-oqgn-tables-del"),
    "extractor-oqgn-tables-ooc": get_model("extractor-oqgn-tables-ooc"),
}


def classify_page(
    model_name: ClassifierModelType, image: ndarray, page_ocr: PageInfo
) -> InferencePageClassification:
    (
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
    )

    return results["boundingBoxes"]

    # return process_extraction_page(
    #     results["boundingBoxes"],
    #     page_ocr,
    #     fields,
    #     mergeable_fields,
    #     tables,
    # )
