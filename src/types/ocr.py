from typing import TypedDict, List


class BoundingBoxCoordinates(TypedDict):
    xMax: float
    xMin: float
    yMax: float
    yMin: float


class BoundingBox(TypedDict):
    confidence: float
    id: str
    coordinates: BoundingBoxCoordinates
    text: str
    blockNumber: int
    lineNumber: int
    pageNumber: int
    wordNumber: int


class PageInfo(TypedDict, total=False):
    confidence: float
    width: float
    height: float
    boundingBoxes: List[BoundingBox]


class Ocr(TypedDict):
    pages: List[PageInfo]
