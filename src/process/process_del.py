from typing import Dict, List
from datetime import datetime
from numpy import ndarray

from src.process.format import FormatConfig, format_bbs, format_csv_output
from src.process.post_process import process_extraction_page, process_table_lines
from src.skwiz.models import extract_page
from src.types.ocr import PageInfo

fields_format: Dict[str, FormatConfig] = {
    "date": {"type": "date", "first": "month", "join_str": " "},
}

line_items_fields_format: Dict[str, FormatConfig] = {
    "day": {"type": "number"},
    "energy": {"type": "number"},
    "hv": {"type": "number"},
    "volume": {"type": "number"},
}


def process_del(
    images: List[ndarray], page_ocr: PageInfo, field_mapping: Dict[str, str]
):
    if len(images) != 1:
        raise ValueError("Expected 1 page for DEL classification")

    csv_output = format_csv_output({}, {}, "")

    extracted_page = extract_page("extractor-oqgn-tables-del", images[0], page_ocr)
    response = process_extraction_page(
        extracted_page,
        page_ocr,
        fields=[f for f in fields_format],
        mergeable_fields=["date"],
        tables=[
            {
                "table": "lineItems",
                "fields": [f for f in line_items_fields_format],
            }
        ],
    )

    fields = {
        field_name: format_bbs(field_value, fields_format[field_name])
        for field_name, field_value in response["fields"].items()
    }

    line_items = process_table_lines(response["tables"]["lineItems"], "day")
    processed_lines = []
    for line in line_items:
        line_response = {}
        for field_name, field_value in line.items():
            format_config = line_items_fields_format[field_name]
            formated_value = format_bbs(field_value, format_config)
            line_response[field_name] = formated_value
        processed_lines.append(line_response)

    base_date = fields.get("date", {}).get("value", None)
    if base_date is None:
        raise ValueError("No date found for DEL")

    date_month = datetime.strptime(base_date, "%Y-%m-%d")

    for line in processed_lines:
        day, energy, hv, volume = (
            line["day"],
            line["energy"],
            line["hv"],
            line["volume"],
        )

        # Skip lines with missing day value
        day_int = None
        try:
            day_int = int(day.get("value", ""))
        except ValueError:
            print(f"Error processing line: {line}", flush=True)
            continue
        except TypeError:
            print(f"Error processing line: {line}", flush=True)
            continue

        # Skip lines with missing energy, hv and volume values
        if energy is None and hv is None and volume is None:
            continue

        date_day = date_month.replace(day=day_int)
        timestamp = date_day.strftime("%Y-%m-%d")
        line_csv_output = format_csv_output(field_mapping, line, timestamp)
        csv_output = csv_output + line_csv_output[1:]

    return csv_output
