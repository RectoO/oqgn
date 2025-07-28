from typing import Optional, TypedDict, List


class Tags(TypedDict):
    number: Optional[bool]
    email: Optional[bool]
    url: Optional[bool]
    percentageSymbol: Optional[bool]
    vatB: Optional[bool]
    vatI: Optional[bool]
    ibanB: Optional[bool]
    ibanI: Optional[bool]
    dateB: Optional[bool]
    dateI: Optional[bool]
    phoneNumberB: Optional[bool]
    phoneNumberI: Optional[bool]
    currencyB: Optional[bool]
    currencyI: Optional[bool]
    amount: Optional[bool]
    percentage: Optional[bool]

class BoundingBoxCoordinates(TypedDict):
    xMax: float
    xMin: float
    yMax: float
    yMin: float


class BoundingBox(TypedDict):
    confidence: float
    id: str
    coordinates: BoundingBoxCoordinates
    tags: Tags
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
