from typing import List
from src.types.ocr import BoundingBoxCoordinates


def normalise_ocr_bounding_box(bbox_coordinates: BoundingBoxCoordinates):
    normalised_coord = [
        bbox_coordinates["xMin"],
        bbox_coordinates["yMin"],
        bbox_coordinates["xMax"],
        bbox_coordinates["yMax"],
    ]
    return [int(1000 * c) for c in normalised_coord]


def contains_bb(parent_bb: BoundingBoxCoordinates, bb: BoundingBoxCoordinates):
    if not parent_bb or not bb:
        return False

    x_min, x_max, y_min, y_max = bb["xMin"], bb["xMax"], bb["yMin"], bb["yMax"]

    middle_x = (x_min + x_max) / 2
    x_inside = middle_x >= parent_bb["xMin"] and middle_x <= parent_bb["xMax"]
    middle_y = (y_min + y_max) / 2
    y_inside = middle_y >= parent_bb["yMin"] and middle_y <= parent_bb["yMax"]

    return x_inside and y_inside


def surrounding_bb(bbs: List[BoundingBoxCoordinates]) -> BoundingBoxCoordinates:
    if len(bbs) == 0:
        return {"xMax": 0, "yMax": 0, "xMin": 0, "yMin": 0}

    x_max = max(bb["xMax"] for bb in bbs)
    y_max = max(bb["yMax"] for bb in bbs)
    x_min = min(bb["xMin"] for bb in bbs)
    y_min = min(bb["yMin"] for bb in bbs)

    return {"xMax": x_max, "yMax": y_max, "xMin": x_min, "yMin": y_min}
