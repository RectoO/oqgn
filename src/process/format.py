import re
from datetime import datetime
from typing import List, Literal, TypedDict
import csv
from src.process.post_process import FieldBoundingBox


class FormatDateConfig(TypedDict):
    type: Literal["date"]
    format: str
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


def format_date(date_str: str, current_format: str, desired_format: str) -> str | None:
    # Parse the original date string into a datetime object
    date_obj = datetime.strptime(date_str.replace(",", ""), current_format)
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
    bbs: List[FieldBoundingBox], join_str: str, current_format: str, desired_format: str
):
    return format_date(get_bbs_text(bbs, join_str), current_format, desired_format)


def format_bbs(bbs: List[FieldBoundingBox], config: FormatConfig):
    if len(bbs) == 0:
        return None
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
                    bbs, config["join_str"], config["format"], "%Y-%m-%d"
                ),
                "rawValue": raw_value,
                "confidence": mean_confidence,
                "ocrConfidence": mean_ocr_confidence,
            }
        else:
            return None
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


def csv_output(output_data: List[List[str]], output_name: str):
    with open(output_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in output_data:
            writer.writerow(row)
