from typing import Dict, List
from numpy import ndarray

from src.ocr.main import ocr_images
from src.utils import FormatConfig, extract_fields, format_csv_output
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
    first_page_ocr: PageInfo,
    stream1_fields_mapping: Dict[str, str],
    stream2_fields_mapping: Dict[str, str],
):
    ocr_pages = [first_page_ocr]
    if len(images) > 1:
        rest_ocr = ocr_images(images[1:])
        ocr_pages.extend(rest_ocr["pages"])

    csv_output = format_csv_output({}, {}, "")
    timestamps: List[str] = []
    for i, (image, ocr_page) in enumerate(zip(images, ocr_pages)):
        extracted_fields = extract_fields(
            model_name="extractor-oqgn",
            image=image,
            page_ocr=ocr_page,
            fields_format=fields_format,
            unmergeable_fields=[],
        )

        timestamp = extracted_fields.get("date", {}).get("value", None)
        if timestamp is None:
            raise ValueError(f"No timestamp found for OXS on page {i+1}")

        stream_value = extracted_fields.get("name", {}).get("value", None)
        if stream_value is None:
            raise ValueError(f"No stream found for OXS on page {i+1}")

        field_mapping = (
            stream1_fields_mapping if "A" in stream_value else stream2_fields_mapping
        )

        timestamps.append(timestamp)
        csv_data = format_csv_output(field_mapping, extracted_fields, timestamp)

        csv_output += csv_data[1:]

    # All page are processed
    page_count = len(images)

    return (min(timestamps), csv_output, page_count)
