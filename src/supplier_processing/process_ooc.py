from typing import Any, Dict, List, Tuple
from numpy import ndarray

from src.ocr.main import ocr_images
from src.utils import FormatConfig, extract_fields, format_csv_output
from src.types.ocr import PageInfo


fields_format: Dict[str, FormatConfig] = {
    "name": {"type": "text", "join_str": " "},
    "date": {"type": "date", "first": "day", "join_str": ""},
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
    "densityStd": {"type": "number"},
    "densityAc": {"type": "number"},
    "hv": {"type": "number"},
}


def process_ooc(
    images: List[ndarray],
    first_page_ocr: PageInfo,
    stream1_fields_mapping: Dict[str, str],
    stream2_fields_mapping: Dict[str, str],
):
    ocr_pages = [first_page_ocr]
    if len(images) > 1:
        rest_ocr = ocr_images(images[1:])
        ocr_pages.extend(rest_ocr["pages"])

    output_by_date_stream: Dict[Tuple[str, str], Dict[str, Any]] = {}
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
            raise ValueError(f"No timestamp found for OOC on page {i+1}")

        stream_value = extracted_fields.get("name", {}).get("value", None)
        if stream_value is None:
            raise ValueError(f"No stream found for OOC on page {i+1}")

        stream_type = "1" if "1" in stream_value else "2"

        if output_by_date_stream.get((timestamp, stream_type), None) is None:
            output_by_date_stream[(timestamp, stream_type)] = extracted_fields
        else:
            for field_name, field_data in extracted_fields.items():
                current_confidence = (
                    output_by_date_stream[(timestamp, stream_type)]
                    .get(field_name, {})
                    .get("confidence", 0)
                )
                new_confidence = field_data.get("confidence", 0)
                if new_confidence > current_confidence:
                    output_by_date_stream[(timestamp, stream_type)][
                        field_name
                    ] = field_data

    timestamps: List[str] = []
    csv_output = format_csv_output({}, {}, "")
    for (timestamp, stream_type), extracted_fields in output_by_date_stream.items():
        field_mapping = (
            stream1_fields_mapping if stream_type == "1" else stream2_fields_mapping
        )
        timestamps.append(timestamp)
        page_csv_output = format_csv_output(field_mapping, extracted_fields, timestamp)
        csv_output = csv_output + page_csv_output[1:]

    # All page are processed
    page_count = len(images)

    return (min(timestamps), csv_output, page_count)
