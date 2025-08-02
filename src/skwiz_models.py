from typing import Any, Dict, Literal, List
from PIL import Image
from numpy import ndarray

from src.skwiz.models import get_model, predict_with_model
from src.types.inference import InferenceBoundingBox, InferencePageClassification
from src.types.ocr import PageInfo


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
) -> InferencePageClassification | None:
    if len(page_ocr["boundingBoxes"]) == 0:
        return None

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


def extract_page(
    model_name: ExtractorModelType,
    image: ndarray,
    page_ocr: PageInfo,
) -> List[InferenceBoundingBox]:
    if len(page_ocr["boundingBoxes"]) == 0:
        return []

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
