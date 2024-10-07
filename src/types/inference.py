from typing import TypedDict, Dict, List
from src.types.ocr import BoundingBoxCoordinates


class DetectedTable(TypedDict):
    header: BoundingBoxCoordinates | None
    rows: BoundingBoxCoordinates | None
    page_index: int


class InferenceBoundingBox(TypedDict):
    id: str
    rawPredictions: Dict[str, Dict[str, float]]


class InferencePageClassification(TypedDict):
    pageNumber: int
    rawPredictions: Dict[str, Dict[str, float]]


class InferenceOutput(TypedDict):
    boundingBoxes: List[InferenceBoundingBox]
    classification: List[InferencePageClassification]
