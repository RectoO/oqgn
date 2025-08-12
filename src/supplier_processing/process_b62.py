from typing import Dict, List

from numpy import ndarray
from src.ocr.main import ocr_images
from src.utils import (
    FormatConfig,
    extract_fields,
    format_csv_output,
)


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
    left_extracted_fields = extract_fields(
        model_name="extractor-oqgn",
        image=left_half,
        page_ocr=left_ocr_page,
        fields_format=left_fields_format,
        mergeable_fields=[],
    )

    # Right
    right_extracted_fields = extract_fields(
        model_name="extractor-oqgn",
        image=right_half,
        page_ocr=right_ocr_page,
        fields_format=right_fields_format,
        mergeable_fields=[],
    )


    timestamp = right_extracted_fields.get("date", {}).get("value", None)
    if timestamp is None or not isinstance(timestamp, str):
        raise ValueError("No timestamp found for B62")

    left_csv_output = format_csv_output(fc1_fields_mapping, left_extracted_fields, timestamp)
    right_csv_output = format_csv_output(fc2_fields_mapping, right_extracted_fields, timestamp)

    # Always 2 page (left and right)
    page_count = 2

    return (timestamp, left_csv_output + right_csv_output[1:], page_count)
