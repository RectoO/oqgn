from typing import Dict, List
from datetime import datetime
from numpy import ndarray

from src.process.format import FormatConfig, format_bbs, get_wide_status
from src.process.post_process import process_extraction_page, process_table_lines
from src.skwiz.models import extract_page
from src.types.ocr import PageInfo

fields_format: Dict[str, FormatConfig] = {
    "date": {"type": "date", "format": "%B %Y", "join_str": " "},
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

    csv_output = [
        ["TimeStamp", "TagName", "Average", "Status", "PercentageOfGoodValues"]
    ]
    date_month = datetime.strptime(fields["date"]["value"], "%Y-%m-%d")

    for line in processed_lines:
        day, energy, hv, volume = (
            line["day"]["value"],
            line["energy"]["value"],
            line["hv"]["value"],
            line["volume"]["value"],
        )
        if day is None or energy is None or hv is None or volume is None:
            continue

        date_day = date_month.replace(day=int(day))
        timestamp = date_day.strftime("%Y-%m-%d")
        for field_name, field_mapping_name in field_mapping.items():
            confidence = (
                line[field_name]["confidence"] + line[field_name]["ocrConfidence"]
            ) / 2
            value = line[field_name]["value"]

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
    # return {
    #     "fields": {
    #         field_name: format_bbs(field_value, fields_format[field_name])
    #         for field_name, field_value in response["fields"].items()
    #     },
    #     "rawLinesItems": response["tables"]["lineItems"],
    #     "processedLinesItems": line_items,
    #     "lineItems": processed_lines,
    # }
    # output = {}
    # for field_name, field_value in response["fields"].items():
    #     format_config = fields_format[field_name]
    #     formated_value = format_bbs(field_value, format_config)
    #     output[field_name] = formated_value

    # for field_name, field_values in response["tables"]["lineItems"].items():
    #     format_config = line_items_fields_format[field_name]
    #     formated_values = [format_bbs(v, format_config) for v in field_values]
    #     output[field_name] = formated_values
