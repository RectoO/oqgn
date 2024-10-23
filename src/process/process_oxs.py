from typing import Dict, List
from numpy import ndarray

from src.process.format import FormatConfig, format_bbs, format_csv_output
from src.process.post_process import process_extraction_page
from src.skwiz.models import extract_page
from src.types.ocr import PageInfo


fields_format: Dict[str, FormatConfig] = {
    "name": {"type": "text", "join_str": " "},
    "date": {"type": "date", "first": "year", "join_str": ""},
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


def process_oxs(
    images: List[ndarray], page_ocr: PageInfo, field_mapping: Dict[str, str]
):
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

    timestamp = output.get("date", {}).get("value", None)
    if timestamp is None:
        raise ValueError("No timestamp found for OXS")

    return format_csv_output(field_mapping, output, timestamp)
