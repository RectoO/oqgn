from typing import Dict, List
from numpy import ndarray

from src.ocr.main import ocr_images
from src.utils import (
    FormatConfig,
    extract_fields,
    format_csv_output,
    process_classification_page,
)
from src.skwiz_models import classify_page
from src.types.ocr import PageInfo


fields_format: Dict[str, FormatConfig] = {
    "name": {"type": "text", "join_str": " "},
    "date": {"type": "date", "first": "day", "join_str": " "},
    "net": {"type": "number"},
    "mass": {"type": "number"},
    "energy": {"type": "number"},
    "hv": {"type": "number"},
    "densityStd": {"type": "number"},
}


def process_lng(
    images: List[ndarray], first_page_ocr: PageInfo, field_mapping: Dict[str, str]
):
    ocr_pages = [first_page_ocr]
    if len(images) > 1:
        rest_ocr = ocr_images(images[1:])
        ocr_pages.extend(rest_ocr["pages"])

    csv_output = format_csv_output({}, {}, "")
    timestamps: List[str] = []
    for i, (image, ocr_page) in enumerate(zip(images, ocr_pages)):
        # Classify page
        classified_page = classify_page(
            model_name="classifier-lng",
            image=image,
            page_ocr=ocr_page,
        )
        classification = (
            process_classification_page(classified_page)
            if classified_page is not None
            else None
        )
        if classification != "GP flow":
            # We skip pages that are not GP flow
            continue

        extracted_fields = extract_fields(
            model_name="extractor-oqgn",
            image=image,
            page_ocr=ocr_page,
            fields_format=fields_format,
            unmergeable_fields=[],
        )
        timestamp = extracted_fields.get("date", {}).get("value", None)
        if timestamp is None:
            raise ValueError(f"No timestamp found for LNG on page {i+1}")

        timestamps.append(timestamp)
        csv_data = format_csv_output(field_mapping, extracted_fields, timestamp)

        csv_output += csv_data[1:]

    if len(timestamps) == 0:
        raise ValueError("No timestamp found for LNG on any pages")

    # All page are processed
    page_count = len(images)

    return (min(timestamps), csv_output, page_count)
