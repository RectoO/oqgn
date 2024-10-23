from typing import Dict, List
from numpy import ndarray

from src.ocr.main import ocr_images
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
    images: List[ndarray],
    page_ocr: PageInfo,
    stream1_fields_mapping: Dict[str, str],
    stream2_fields_mapping: Dict[str, str],
):
    if len(images) > 2:
        raise ValueError("Expected max 2 page for OXS classification")

    csv_output = format_csv_output({}, {}, "")

    for image_index, image in enumerate(images):
        if image_index == 0:
            image_ocr = page_ocr
        else:
            image_ocr = ocr_images([image])["pages"][0]

        extracted_page = extract_page("extractor-oqgn", image, image_ocr)
        response = process_extraction_page(
            extracted_page,
            image_ocr,
            fields=[f for f in fields_format],
            mergeable_fields=["name"],
            tables=[],
        )

        output = {}
        for field_name, field_value in response["fields"].items():
            format_config = fields_format[field_name]
            formated_value = format_bbs(field_value, format_config)
            output[field_name] = formated_value

        stream_value = output.get("name", {}).get("value", None)
        if stream_value is None:
            raise ValueError(f"No stream found for OXS at page {image_index + 1}")

        field_mapping = (
            stream1_fields_mapping if "A" in stream_value else stream2_fields_mapping
        )

        timestamp = output.get("date", {}).get("value", None)
        if timestamp is None:
            raise ValueError("No timestamp found for OXS")

        page_csv_output = format_csv_output(field_mapping, output, timestamp)
        csv_output = csv_output + page_csv_output[1:]
    return csv_output
