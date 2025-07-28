from typing import Any, Dict, Callable, List, Tuple, TypedDict
from src.skwiz.bounding_boxes import contains_bb, surrounding_bb
from src.types.inference import InferenceBoundingBox, InferencePageClassification
from src.types.ocr import BoundingBoxCoordinates, PageInfo

THRESHOLD_TO_MERGE = 0.5
MIN_FIELD_PRED_THRESHOLD = 0.001
MIN_TABLE_FIELD_PRED_THRESHOLD = 0.5


def get_max_key(data: Dict[str, float]) -> str:
    key_func: Callable[[str], float] = lambda k: data[k]
    return max(data, key=key_func)


def get_max_key_value(data: Dict[str, float]) -> Tuple[str, float]:
    key_func: Callable[[str], float] = lambda k: data[k]
    key = max(data, key=key_func)
    return key, data[key]


def process_classification_page(classification: InferencePageClassification):
    raw_predictions: Dict[str, float] = classification.get("rawPredictions", {}).get(
        "_classification", {}
    )

    # Get the top prediction
    top_prediction, score = get_max_key_value(raw_predictions)

    print(f"Classification: {top_prediction}, score: {score}", flush=True)

    if top_prediction == "None":
        return None

    if top_prediction[0] == "_":
        return top_prediction[1:]
    return top_prediction


def get_fields_prediction(
    bb_pred: InferenceBoundingBox, fields: List[str]
) -> Dict[str, float]:
    pred_fields = bb_pred.get("rawPredictions", {}).get("fields", {})
    return {field: pred_fields.get(f"_{field}", 0.0) for field in fields}


def get_table_fields_prediction(
    bb_pred: InferenceBoundingBox, table: str, fields: List[str]
) -> Dict[str, float]:
    table_columns = bb_pred.get("rawPredictions", {}).get(f"__{table}_columns", {})
    table_rows = bb_pred.get("rawPredictions", {}).get(f"__{table}_rows", {})
    row_pred = get_max_key(table_rows)

    if row_pred != "line":
        return {field: 0.0 for field in fields}

    return {field: table_columns.get(f"_{field}", 0.0) for field in fields}


class FieldBoundingBox(TypedDict):
    id: str
    coordinates: BoundingBoxCoordinates
    text: str
    blockNumber: int
    lineNumber: int
    wordNumber: int
    pageNumber: int
    prediction: str
    confidence: float
    ocr_confidence: float


def can_merge_bbs(
    mergeable_fields: List[str],
    current_bbs: List[FieldBoundingBox],
    bb2: FieldBoundingBox,
    field: str,
    all_bb_with_predictions,
) -> bool:

    if len(current_bbs) == 0:
        return True

    if field not in mergeable_fields:
        return False

    if current_bbs[0]["lineNumber"] != bb2["lineNumber"]:
        return False

    bbs_coordinates = surrounding_bb([bb["coordinates"] for bb in current_bbs + [bb2]])
    resulting_bbs = [
        bb
        for bb in all_bb_with_predictions
        if contains_bb(bbs_coordinates, bb["coordinates"])
    ]
    resulting_bbs_field_confidence = [bb["predictions"][field] for bb in resulting_bbs]
    mean_confidence = sum(resulting_bbs_field_confidence) / len(
        resulting_bbs_field_confidence
    )

    if mean_confidence < THRESHOLD_TO_MERGE:
        return False

    return True


class TablesConfig(TypedDict):
    table: str
    fields: List[str]


class ExtractionPage(TypedDict):
    fields: Dict[str, List[FieldBoundingBox]]
    tables: Dict[str, Dict[str, List[FieldBoundingBox]]]


def process_extraction_page(
    bounding_boxes: List[InferenceBoundingBox],
    page_ocr: PageInfo,
    fields: List[str],
    mergeable_fields: List[str],
    tables: List[TablesConfig],
) -> ExtractionPage:
    # Merge ocr and predictions
    prediction_bbs_by_id = {bb["id"]: bb for bb in bounding_boxes}
    bb_with_predictions: List[Any] = [
        {
            "id": bb["id"],
            "confidence": bb["confidence"],
            "coordinates": bb["coordinates"],
            "text": bb["text"],
            "blockNumber": bb["blockNumber"],
            "lineNumber": bb["lineNumber"],
            "wordNumber": bb["wordNumber"],
            "pageNumber": bb["pageNumber"],
            "predictions": get_fields_prediction(
                prediction_bbs_by_id[bb["id"]], fields
            ),
            "tablePredictions": {
                table["table"]: get_table_fields_prediction(
                    prediction_bbs_by_id[bb["id"]], table["table"], table["fields"]
                )
                for table in tables
            },
        }
        for bb in page_ocr["boundingBoxes"]
    ]

    # Fields
    prediction_for_all_fields: List[FieldBoundingBox] = []

    for bb in bb_with_predictions:
        for field, pred_value in bb["predictions"].items():
            if pred_value >= MIN_FIELD_PRED_THRESHOLD:
                prediction_for_all_fields.append(
                    {
                        "id": bb["id"],
                        "coordinates": bb["coordinates"],
                        "text": bb["text"],
                        "blockNumber": bb["blockNumber"],
                        "lineNumber": bb["lineNumber"],
                        "wordNumber": bb["wordNumber"],
                        "pageNumber": bb["pageNumber"],
                        "ocr_confidence": bb["confidence"],
                        "prediction": field,
                        "confidence": pred_value,
                    }
                )

    sorted_prediction_for_all_fields: List[FieldBoundingBox] = sorted(
        prediction_for_all_fields, key=lambda x: x["confidence"], reverse=True
    )
    field_preds: Dict[str, List[FieldBoundingBox]] = {field: [] for field in fields}
    used_bb_ids = set()
    for bb in sorted_prediction_for_all_fields:
        bb_id = bb["id"]
        if bb_id in used_bb_ids:
            continue

        prediction = bb["prediction"]
        if can_merge_bbs(
            mergeable_fields,
            field_preds[prediction],
            bb,
            prediction,
            bb_with_predictions,
        ):
            field_preds[prediction].append(bb)
            used_bb_ids.add(bb_id)

    # Tables
    table_preds = {}
    for table in tables:
        table_prediction_for_all_fields: List[FieldBoundingBox] = []

        for bb in bb_with_predictions:
            for field, pred_value in bb["tablePredictions"][table["table"]].items():
                if pred_value >= MIN_TABLE_FIELD_PRED_THRESHOLD:
                    table_prediction_for_all_fields.append(
                        {
                            "id": bb["id"],
                            "coordinates": bb["coordinates"],
                            "text": bb["text"],
                            "blockNumber": bb["blockNumber"],
                            "lineNumber": bb["lineNumber"],
                            "wordNumber": bb["wordNumber"],
                            "pageNumber": bb["pageNumber"],
                            "ocr_confidence": bb["confidence"],
                            "prediction": field,
                            "confidence": pred_value,
                        }
                    )

        sorted_table_prediction_for_all_fields: List[FieldBoundingBox] = sorted(
            table_prediction_for_all_fields, key=lambda x: x["confidence"], reverse=True
        )
        table_field_preds: Dict[str, List[FieldBoundingBox]] = {
            field: [] for field in table["fields"]
        }
        used_bb_ids = set()
        for bb in sorted_table_prediction_for_all_fields:
            bb_id = bb["id"]
            if bb_id in used_bb_ids:
                continue

            prediction = bb["prediction"]
            table_field_preds[prediction].append(bb)
            used_bb_ids.add(bb_id)

        table_preds[table["table"]] = table_field_preds

    return {
        "fields": field_preds,
        "tables": table_preds,
    }


def process_table_lines(
    table_field_preds: dict[str, List[FieldBoundingBox]], align_column_name: str
):
    align_column_predictions = table_field_preds[align_column_name]

    lines: list[dict[str, List[FieldBoundingBox]]] = []
    for pred in sorted(
        align_column_predictions,
        key=lambda x: (x["coordinates"]["yMin"] + x["coordinates"]["yMax"]) / 2,
    ):
        y_min = pred["coordinates"]["yMin"]
        y_max = pred["coordinates"]["yMax"]

        line_pred: dict[str, List[FieldBoundingBox]] = {align_column_name: [pred]}
        for field_name, predictions in table_field_preds.items():
            if predictions == align_column_name:
                continue

            line_field_predictions = []
            for bb in predictions:
                if (
                    y_min
                    <= (bb["coordinates"]["yMin"] + bb["coordinates"]["yMax"]) / 2
                    <= y_max
                ):
                    line_field_predictions.append(bb)

            line_pred[field_name] = line_field_predictions
        lines.append(line_pred)

    return lines
