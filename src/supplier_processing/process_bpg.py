from typing import Any, Dict, List
from numpy import ndarray

from src.ocr.main import ocr_images
from src.utils import FormatConfig, extract_fields, format_csv_output
from src.types.ocr import PageInfo


fields_format: Dict[str, FormatConfig] = {
    "name": {"type": "text", "join_str": " "},
    "date": {"type": "date", "first": "day", "join_str": ""},
    "pressure1": {"type": "number"},
    "temperature1": {"type": "number"},
    "hv": {"type": "number"},
    "densityAc": {"type": "number"},
    "densityStd": {"type": "number"},
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

def process_bpg(
    images: List[ndarray], first_page_ocr: PageInfo, field_mapping: Dict[str, str]
):
    ocr_pages = [first_page_ocr]
    if len(images) > 1:
        rest_ocr = ocr_images(images[1:])
        ocr_pages.extend(rest_ocr["pages"])

    output_by_date: Dict[str, Dict[str, Any]] = {}
    for i, (image, ocr_page) in enumerate(zip(images, ocr_pages)):
        extracted_fields = extract_fields(
            model_name="extractor-oqgn",
            image=image,
            page_ocr=ocr_page,
            fields_format=fields_format,
            mergeable_fields=[],
        )
        timestamp = extracted_fields.get("date", {}).get("value", None)
        if timestamp is None:
            raise ValueError(f"No timestamp found for BPG on page {i+1}")

        if output_by_date.get(timestamp, None) is None:
            output_by_date[timestamp] = extracted_fields
        else:
            for field_name, field_data in extracted_fields.items():
                current_confidence = (
                    output_by_date[timestamp].get(field_name, {}).get("confidence", 0)
                )
                new_confidence = field_data.get("confidence", 0)
                if new_confidence > current_confidence:
                    output_by_date[timestamp][field_name] = field_data

    csv_output = format_csv_output({}, {}, "")
    timestamps = []
    for timestamp, extracted_fields in output_by_date.items():
        timestamps.append(timestamp)
        csv_data = format_csv_output(field_mapping, extracted_fields, timestamp)
        csv_output += csv_data[1:]

    return (min(timestamps), csv_output)

