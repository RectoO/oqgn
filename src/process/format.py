import re
from typing import List, Literal, TypedDict, Dict, Any
import csv
import dateutil.parser as date_parser
from src.process.post_process import FieldBoundingBox


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
    csv_data = [["TimeStamp", "TagName", "Average", "Status", "PercentageOfGoodValues", "ocrConfidence", "confidence"]]

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
                field_data.get("ocrConfidence", 0),
                field_data.get("confidence", 0),
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
