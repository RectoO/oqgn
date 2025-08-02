from typing import Dict, List
from numpy import ndarray

from src.utils import FormatConfig, format_csv_output, extract_fields
from src.types.ocr import PageInfo


fields_format: Dict[str, FormatConfig] = {
    "name": {"type": "text", "join_str": " "},
    "date": {"type": "date", "first": "month", "join_str": ""},
    "gross": {"type": "number"},
    "net": {"type": "number"},
    "mass": {"type": "number"},
    "energy": {"type": "number"},
    "pressure1": {"type": "number"},
    "pressure2": {"type": "number"},
    "temperature1": {"type": "number"},
    "temperature2": {"type": "number"},
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


# First page processed
def process_ara(
    images: List[ndarray], page_ocr: PageInfo, field_mapping: Dict[str, str]
):
    extracted_fields = extract_fields(
        model_name="extractor-oqgn",
        image=images[0],
        page_ocr=page_ocr,
        fields_format=fields_format,
        mergeable_fields=[],
    )

    timestamp = extracted_fields.get("date", {}).get("value", None)
    if timestamp is None:
        raise ValueError("No timestamp found for ARA")

    return (timestamp, format_csv_output(field_mapping, extracted_fields, timestamp))
