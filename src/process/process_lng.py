from typing import Dict, List

from numpy import ndarray

from src.ocr.main import ocr_images
from src.process.format import FormatConfig, format_bbs, get_wide_status
from src.process.post_process import (
    process_classification_page,
    process_extraction_page,
)
from src.skwiz.models import classify_page, extract_page
from src.types.ocr import PageInfo


fields_format: Dict[str, FormatConfig] = {
    "name": {"type": "text", "join_str": " "},
    "date": {"type": "date", "format": "%d %b %Y", "join_str": " "},
    "net": {"type": "number"},
    "mass": {"type": "number"},
    "energy": {"type": "number"},
    "hv": {"type": "number"},
    "densityStd": {"type": "number"},
}


def process_lng(
    images: List[ndarray], page_ocr: PageInfo, field_mapping: Dict[str, str]
):
    csv_output = [
        ["TimeStamp", "TagName", "Average", "Status", "PercentageOfGoodValues"]
    ]
    for image_index, image in enumerate(images):
        if image_index == 0:
            image_ocr = page_ocr
        else:
            image_ocr = ocr_images([image])["pages"][0]

        # Classify page
        classified_page = classify_page(
            model_name="classifier-lng",
            image=images[0],
            page_ocr=image_ocr,
        )
        classification = process_classification_page(classified_page)

        if classification != "GP flow":
            # We skip pages that are not GP flow
            continue
        extracted_page = extract_page("extractor-oqgn", image, image_ocr)
        response = process_extraction_page(
            extracted_page,
            image_ocr,
            fields=[f for f in fields_format],
            mergeable_fields=["date", "name"],
            tables=[],
        )

        output = {}
        for field_name, field_value in response["fields"].items():
            format_config = fields_format[field_name]
            formated_value = format_bbs(field_value, format_config)
            output[field_name] = formated_value

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
