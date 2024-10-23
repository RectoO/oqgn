from typing import Dict, List
from numpy import ndarray

from src.process.format import FormatConfig, format_bbs, get_wide_status
from src.process.post_process import process_extraction_page
from src.skwiz.models import extract_page
from src.types.ocr import PageInfo


fields_format: Dict[str, FormatConfig] = {
    "name": {"type": "text", "join_str": " "},
    "date": {"type": "date", "format": "%d/%m/%Y", "join_str": ""},
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


def process_bpk(
    images: List[ndarray], page_ocr: PageInfo, field_mapping: Dict[str, str]
):
    if len(images) != 1:
        raise ValueError("Expected 1 page for BPK classification")

    extracted_page = extract_page("extractor-oqgn", images[0], page_ocr)
    response = process_extraction_page(
        extracted_page,
        page_ocr,
        fields=[f for f in fields_format],
        mergeable_fields=["name"],
        tables=[],
    )

    output = {}
    for field_name, field_value in response["fields"].items():
        format_config = fields_format[field_name]
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
