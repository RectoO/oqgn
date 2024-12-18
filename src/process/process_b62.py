from typing import Dict, List

from numpy import ndarray
from src.ocr.main import ocr_images
from src.process.format import (
    FormatConfig,
    format_bbs,
    format_csv_output,
)
from src.process.post_process import process_extraction_page
from src.skwiz.models import extract_page


left_fields_format: Dict[str, FormatConfig] = {
    "gross": {"type": "number"},
    "net": {"type": "number"},
    "mass": {"type": "number"},
    "energy": {"type": "number"},
    "pressure1": {"type": "number"},
    "temperature1": {"type": "number"},
    "c1": {"type": "number"},
    "c2": {"type": "number"},
    "c3": {"type": "number"},
    "ic4": {"type": "number"},
    "nc4": {"type": "number"},
    "ic5": {"type": "number"},
    "nc5": {"type": "number"},
    "c6": {"type": "number"},
    "n2": {"type": "number"},
    "co2": {"type": "number"},
    "hv": {"type": "number"},
    "densityAc": {"type": "number"},
    "densityStd": {"type": "number"},
}

right_fields_format: Dict[str, FormatConfig] = {
    "date": {"type": "date", "first": "month", "join_str": ""},
    "gross": {"type": "number"},
    "net": {"type": "number"},
    "mass": {"type": "number"},
    "energy": {"type": "number"},
    "pressure1": {"type": "number"},
    "temperature1": {"type": "number"},
    "c1": {"type": "number"},
    "c2": {"type": "number"},
    "c3": {"type": "number"},
    "ic4": {"type": "number"},
    "nc4": {"type": "number"},
    "ic5": {"type": "number"},
    "nc5": {"type": "number"},
    "c6": {"type": "number"},
    "n2": {"type": "number"},
    "co2": {"type": "number"},
    "hv": {"type": "number"},
    "densityAc": {"type": "number"},
    "densityStd": {"type": "number"},
}


def process_b62(
    images: List[ndarray],
    fc1_fields_mapping: Dict[str, str],
    fc2_fields_mapping: Dict[str, str],
):
    # Only one page
    if len(images) != 1:
        raise ValueError("Expected 1 pages for B62 classification")

    image = images[0]
    # Split in two halves
    width = image.shape[1]
    mid = width // 2
    left_half = image[:, :mid, :]
    right_half = image[:, mid:, :]

    # Perform OCR
    ocr_results = ocr_images([left_half, right_half])
    left_ocr_page = ocr_results["pages"][0]
    right_ocr_page = ocr_results["pages"][1]

    # Left
    left_extracted_page = extract_page("extractor-oqgn", left_half, left_ocr_page)
    left_response = process_extraction_page(
        left_extracted_page,
        left_ocr_page,
        fields=[f for f in left_fields_format],
        mergeable_fields=[],
        tables=[],
    )

    # Right
    right_extracted_page = extract_page("extractor-oqgn", right_half, right_ocr_page)
    right_response = process_extraction_page(
        right_extracted_page,
        right_ocr_page,
        fields=[f for f in right_fields_format],
        mergeable_fields=[],
        tables=[],
    )

    left_output = {}
    for field_name, field_value in left_response["fields"].items():
        format_config = left_fields_format[field_name]
        formated_value = format_bbs(field_value, format_config)
        left_output[field_name] = formated_value

    right_output = {}
    for field_name, field_value in right_response["fields"].items():
        format_config = right_fields_format[field_name]
        formated_value = format_bbs(field_value, format_config)
        right_output[field_name] = formated_value

    timestamp = right_output.get("date", {}).get("value", None)
    if timestamp is None:
        raise ValueError("No timestamp found for B62")

    left_csv_output = format_csv_output(fc1_fields_mapping, left_output, timestamp)
    right_csv_output = format_csv_output(fc2_fields_mapping, right_output, timestamp)

    return left_csv_output + right_csv_output[1:]
