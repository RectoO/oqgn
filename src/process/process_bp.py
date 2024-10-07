from typing import Dict, List
from numpy import ndarray

from src.ocr.main import ocr_images
from src.process.format import FormatConfig, format_bbs, get_wide_status
from src.process.post_process import process_extraction_page
from src.skwiz.models import extract_page
from src.types.ocr import PageInfo


first_page_fields_format: Dict[str, FormatConfig] = {
    "name": {"type": "text", "join_str": " "},
    "date": {"type": "date", "format": "%d/%b/%Y", "join_str": ""},
    "pressure1": {"type": "number"},
    "temperature1": {"type": "number"},
    "hv": {"type": "number"},
    "densityAc": {"type": "number"},
    "densityStd": {"type": "number"},
}

second_page_fields_format: Dict[str, FormatConfig] = {
    "gross": {"type": "number"},
    "net": {"type": "number"},
    "mass": {"type": "number"},
    "energy": {"type": "number"},
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
}


def process_bp(
    images: List[ndarray], first_page_ocr: PageInfo, field_mapping: Dict[str, str]
):
    if len(images) != 2:
        raise ValueError("Expected 2 pages for BP classification")

    second_page_ocr_result = ocr_images(images[1:2])
    second_page_ocr = second_page_ocr_result["pages"][0]

    # First page
    first_extracted_page = extract_page("extractor-oqgn", images[0], first_page_ocr)
    first_page_response = process_extraction_page(
        first_extracted_page,
        first_page_ocr,
        fields=[f for f in first_page_fields_format],
        mergeable_fields=[],
        tables=[],
    )
    # Second page
    second_extracted_page = extract_page("extractor-oqgn", images[1], second_page_ocr)
    second_page_response = process_extraction_page(
        second_extracted_page,
        second_page_ocr,
        fields=[f for f in second_page_fields_format],
        mergeable_fields=[],
        tables=[],
    )

    output = {}
    for field_name, field_value in first_page_response["fields"].items():
        format_config = first_page_fields_format[field_name]
        formated_value = format_bbs(field_value, format_config)
        output[field_name] = formated_value
    for field_name, field_value in second_page_response["fields"].items():
        format_config = second_page_fields_format[field_name]
        formated_value = format_bbs(field_value, format_config)
        output[field_name] = formated_value

    csv_output = [
        ["TimeStamp", "TagName", "Average", "Status", "PercentageOfGoodValues"]
    ]

    timestamp = output["date"]["value"]
    for field_name, field_mapping_name in field_mapping.items():
        confidence = (
            output[field_name]["confidence"] + output[field_name]["ocrConfidence"]
        ) / 2
        value = output[field_name]["value"]

        csv_output.append(
            [
                timestamp,
                field_mapping_name,
                value,
                get_wide_status(value, confidence),
                confidence,
            ]
        )

    return csv_output