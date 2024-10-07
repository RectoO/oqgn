from typing import Dict, List

from numpy import ndarray

from src.ocr.main import ocr_images
from src.process.format import FormatConfig, format_bbs, get_bbs_text, get_wide_status
from src.process.post_process import process_extraction_page, process_table_lines
from src.skwiz.models import extract_page
from src.types.ocr import PageInfo

fields_format: Dict[str, FormatConfig] = {
    "stream": {"type": "text", "join_str": " "},
    "date": {"type": "date", "format": "%d/%m/%Y", "join_str": ""},
}

quantity_table_fields_format: Dict[str, FormatConfig] = {
    "description": {"type": "text", "join_str": ""},
    "gross": {"type": "number"},
    "net": {"type": "number"},
    "mass": {"type": "number"},
    "energy": {"type": "number"},
    "pressure": {"type": "number"},
    "temperature": {"type": "number"},
}

quality_table_fields_format: Dict[str, FormatConfig] = {
    "description": {"type": "text", "join_str": ""},
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
    page_ocr: PageInfo,
    stream1_fields_mapping: Dict[str, str],
    stream2_fields_mapping: Dict[str, str],
):
    if len(images) != 4:
        raise ValueError("Expected 4 page for OOC classification")

    csv_output = [
        ["TimeStamp", "TagName", "Average", "Status", "PercentageOfGoodValues"]
    ]
    for image_index, image in enumerate(images):
        if image_index == 0:
            image_ocr = page_ocr
        else:
            image_ocr = ocr_images([image])["pages"][0]

        extracted_page = extract_page("extractor-oqgn-tables-ooc", image, image_ocr)
        response = process_extraction_page(
            extracted_page,
            image_ocr,
            fields=[f for f in fields_format],
            mergeable_fields=["stream"],
            tables=[
                {
                    "table": "quantityTable",
                    "fields": [f for f in quantity_table_fields_format],
                },
                {
                    "table": "qualityTable",
                    "fields": [f for f in quality_table_fields_format],
                },
            ],
        )

        output = {}
        for field_name, field_value in response["fields"].items():
            format_config = fields_format[field_name]
            formated_value = format_bbs(field_value, format_config)
            output[field_name] = formated_value

        # Quality table
        quality_table_output = {}
        for field_name, field_values in response["tables"]["qualityTable"].items():
            if len(field_values) == 0:
                continue
            format_config = quality_table_fields_format[field_name]
            # Get the last prediction
            last_pred = max(field_values, key=lambda x: x["coordinates"]["yMax"])
            formated_values = format_bbs([last_pred], format_config)
            quality_table_output[field_name] = formated_values

        # Quantity table
        quantity_table_output = {}
        quantity_table = process_table_lines(
            response["tables"]["quantityTable"], "description"
        )
        for line in quantity_table:
            description = get_bbs_text(line["description"], " ")

            if description.lower() == "total":
                for field_name in ["gross", "net", "mass", "energy"]:
                    quantity_table_output[field_name] = format_bbs(
                        line[field_name], quantity_table_fields_format[field_name]
                    )

            if description.lower() == "average":
                for field_name in ["pressure", "temperature"]:
                    quantity_table_output[field_name] = format_bbs(
                        line[field_name], quantity_table_fields_format[field_name]
                    )

        field_mapping = (
            stream1_fields_mapping
            if "1" in output["stream"]["value"]
            else stream2_fields_mapping
        )
        timestamp = output["date"]["value"]
        data = quantity_table_output if quantity_table_output else quality_table_output
        for field_name, field_mapping_name in field_mapping.items():
            raw_data = data.get(field_name, {})
            if not raw_data:
                continue
            confidence = (raw_data["confidence"] + raw_data["ocrConfidence"]) / 2
            value = raw_data["value"]

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
