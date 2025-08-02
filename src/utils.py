import os
import re
import csv
import shutil
import dateutil.parser as date_parser
from typing import Any, Dict, Callable, List, Tuple, TypedDict, Literal
from numpy import ndarray

from src.skwiz.bounding_boxes import contains_bb, surrounding_bb
from src.types.inference import InferenceBoundingBox, InferencePageClassification
from src.types.ocr import BoundingBoxCoordinates, PageInfo
from src.constants import OUTPUT_FOLDER, TMP_FOLDER
from src.skwiz_models import ExtractorModelType, extract_page


THRESHOLD_TO_MERGE = 0.5
MIN_FIELD_PRED_THRESHOLD = 0.001
MIN_TABLE_FIELD_PRED_THRESHOLD = 0.5


class FormatDateConfig(TypedDict):
    type: Literal["date"]
    first: Literal["day", "month", "year"]
    join_str: str


class FormatNumberConfig(TypedDict):
    type: Literal["number"]


class FormatTextConfig(TypedDict):
    type: Literal["text"]
    join_str: str


FormatConfig = FormatDateConfig | FormatNumberConfig | FormatTextConfig



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
    top_prediction, _score = get_max_key_value(raw_predictions)

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



def format_number(raw_text: str) -> float | None:
    def unformat_number(number_str: str) -> float:
        comma_last_index = number_str.rfind(",")
        dot_last_index = number_str.rfind(".")
        used_separator = "," if comma_last_index > dot_last_index else "."
        # Remove the separator and convert the string to a float
        if used_separator == ",":
            number_str = number_str.replace(".", "").replace(",", ".")
        else:
            number_str = number_str.replace(",", "")

        # Keep only last dot
        if number_str.count(".") > 1:
            number_str = number_str.replace(".", "", number_str.count(".") - 1)

        return float(number_str)

    # Remove trailing commas or dots
    text = re.sub(r"[.,]$", "", raw_text)

    # Find the first matched number
    # match = re.findall(r"-?(\d+(\.|,)?)+", text.replace(" ", ""))
    matches = re.finditer(r"-?(\d+(\.|,)?)+", text.replace(" ", ""))
    for match in matches:
        number = match[0]
        return unformat_number(number)

    return None


def format_date(
    date_str: str, first: Literal["day", "month", "year"], desired_format: str
) -> str | None:
    # Parse the original date string into a datetime object
    date_obj = date_parser.parse(
        date_str,
        dayfirst=True if first == "day" else False,
        yearfirst=True if first == "year" else False,
    )

    # Format the datetime object into the desired format
    return date_obj.strftime(desired_format)


def get_bbs_text(bbs: List[FieldBoundingBox], join_str: str = ""):
    sorted_bbs = sorted(bbs, key=lambda x: x["coordinates"]["xMin"])
    return (join_str or "").join([bb["text"] for bb in sorted_bbs])


def format_bbs_text(bbs: List[FieldBoundingBox]):
    return get_bbs_text(bbs, " ")


def format_bbs_number(bbs: List[FieldBoundingBox]):
    return format_number(get_bbs_text(bbs))


def format_bbs_date(
    bbs: List[FieldBoundingBox],
    join_str: str,
    first: Literal["day", "month", "year"],
    desired_format: str,
):
    return format_date(get_bbs_text(bbs, join_str), first, desired_format)


def format_bbs(bbs: List[FieldBoundingBox], config: FormatConfig):
    if len(bbs) == 0:
        return {}
    all_confidence = [bb["confidence"] for bb in bbs]
    mean_confidence = sum(all_confidence) / len(all_confidence)

    all_ocr_confidence = [bb["ocr_confidence"] for bb in bbs]
    mean_ocr_confidence = sum(all_ocr_confidence) / len(all_ocr_confidence)
    raw_value = get_bbs_text(bbs, " ")

    try:
        if config["type"] == "text":
            return {
                "value": format_bbs_text(bbs),
                "rawValue": raw_value,
                "confidence": mean_confidence,
                "ocrConfidence": mean_ocr_confidence,
            }
        elif config["type"] == "number":
            return {
                "value": format_bbs_number(bbs),
                "rawValue": raw_value,
                "confidence": mean_confidence,
                "ocrConfidence": mean_ocr_confidence,
            }
        elif config["type"] == "date":
            return {
                "value": format_bbs_date(
                    bbs, config["join_str"], config["first"], "%Y-%m-%d"
                ),
                "rawValue": raw_value,
                "confidence": mean_confidence,
                "ocrConfidence": mean_ocr_confidence,
            }
    except Exception as _e:
        return {
            "error": "Failed to format",
            "value": None,
            "confidence": 0.0,
            "ocrConfidence": 0.0,
            "rawValue": get_bbs_text(bbs, " "),
        }


def get_wide_status(value, confidence):
    if value is None or value == "":
        return "NoData"
    if confidence < 0.05:
        return "Error"
    if confidence > 0.5:
        return "OK"
    return "OKWithErrors"


def format_csv_output(
    field_mapping: Dict[str, str], data: Dict[str, Dict[str, Any]], timestamp: str
):
    csv_data = [["TimeStamp", "TagName", "Average", "Status", "PercentageOfGoodValues"]]

    for field_name, field_mapping_name in field_mapping.items():
        field_data = data.get(field_name, {}) or {}
        confidence = min(
            field_data.get("confidence", 0),
            field_data.get("ocrConfidence", 0),
        )
        value = field_data.get("value", None)

        csv_data.append(
            [
                timestamp,
                field_mapping_name,
                "" if value is None else value,
                get_wide_status(value, confidence),
                confidence,
            ]
        )

    return csv_data


def update_tag_default_values(csv_data, tag_default_values):
    header = csv_data[0]
    return [
        header,
        *[
            (
                [
                    row[0],
                    row[1],
                    tag_default_values.get(row[1], row[2]),
                    row[3],
                    row[4],
                ]
                if row[3] == "NoData"
                else row
            )
            for row in csv_data[1:]
        ],
    ]


def csv_output(output_data: List[List[str]], output_name: str):
    with open(output_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in output_data:
            writer.writerow(row)


def csv_output_tmp_move(
    original_file_name: str,
    classification: str,
    date: str,
    data: List[List[str]],
):
    # Output file name
    base_file_name = ".".join(original_file_name.split('.')[:-1])
    formatted_date = "".join(date.split('-'))
    output_file_name = f"{classification}_{formatted_date}_{base_file_name}.csv"
    # Create tmp file
    tmp_file_path = os.path.join(TMP_FOLDER, output_file_name)
    with open(tmp_file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in data:
            writer.writerow(row)

    # Output file path
    output_file_path = os.path.join(OUTPUT_FOLDER, output_file_name)
    shutil.move(tmp_file_path, output_file_path)


def extract_fields(
    model_name: ExtractorModelType,
    image: ndarray,
    page_ocr: PageInfo,
    fields_format: Dict[str, FormatConfig],
    mergeable_fields: List[str],
):
    extracted_page = extract_page(model_name, image, page_ocr)
    processed_extracted_page = process_extraction_page(
        extracted_page,
        page_ocr,
        fields=[f for f in fields_format],
        mergeable_fields=mergeable_fields,
        tables=[],
    )

    return {
        field_name: format_bbs(field_value, fields_format[field_name])
        for field_name, field_value in processed_extracted_page["fields"].items()
    }
