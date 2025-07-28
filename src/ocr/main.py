from typing import List
from numpy import ndarray
import torch
from doctr.models import ocr_predictor, fast_base, crnn_vgg16_bn
from doctr.io import DocumentFile
from src.types.ocr import Ocr, PageInfo, BoundingBox


det_model = fast_base(pretrained=False, pretrained_backbone=False)
det_params = torch.load("/var/www/models/ocr/fast_base-688a8b34.pt", map_location="cpu")
det_model.load_state_dict(det_params)

reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
reco_params = torch.load(
    "/var/www/models/ocr/crnn_vgg16_bn-9762b0b0.pt", map_location="cpu"
)
reco_model.load_state_dict(reco_params)

model = ocr_predictor(
    det_arch=det_model,
    reco_arch=reco_model,
    pretrained=False,
    assume_straight_pages=True,
    preserve_aspect_ratio=True,
    detect_orientation=False,
    # detect_language=True,
)

def transform_doctr_ocr(doctr_ocr) -> Ocr:
    def generate_bounding_box_id(page_number, coordinates):
        return f"{page_number}_{coordinates['yMin']}-{coordinates['yMax']}-{coordinates['xMin']}-{coordinates['xMax']}".replace(
            ".", ""
        )

    def process_page(page, page_index) -> PageInfo:
        page_bounding_boxes: List[BoundingBox] = []

        height, width = page["dimensions"]
        block_index = 0
        line_index = 0
        word_index = 0
        for block in page["blocks"]:
            for line in block["lines"]:
                for word in line["words"]:
                    value, geometry, confidence = (
                        word["value"],
                        word["geometry"],
                        word["confidence"],
                    )
                    (x_min, y_min), (x_max, y_max) = geometry

                    if not value:
                        continue

                    bounding_box: BoundingBox = {
                        "confidence": confidence,
                        "id": generate_bounding_box_id(
                            page_index + 1,
                            {
                                "xMin": x_min,
                                "yMin": y_min,
                                "xMax": x_max,
                                "yMax": y_max,
                            },
                        ),
                        "tags": {
                            "number": False,
                            "email": False,
                            "url": False,
                            "percentageSymbol": False,
                            "vatB": False,
                            "vatI": False,
                            "ibanB": False,
                            "ibanI": False,
                            "dateB": False,
                            "dateI": False,
                            "phoneNumberB": False,
                            "phoneNumberI": False,
                            "currencyB": False,
                            "currencyI": False,
                            "amount": False,
                            "percentage": False,
                        },
                        "coordinates": {
                            "xMin": x_min,
                            "yMin": y_min,
                            "xMax": x_max,
                            "yMax": y_max,
                        },
                        "text": value,
                        "blockNumber": block_index + 1,
                        "lineNumber": line_index + 1,
                        "wordNumber": word_index + 1,
                        "pageNumber": page_index + 1,
                    }

                    page_bounding_boxes.append(bounding_box)
                    word_index += 1
                line_index += 1
            block_index += 1

        return {
            "width": width,
            "height": height,
            "boundingBoxes": page_bounding_boxes,
        }

    return {
        "pages": [process_page(page, i) for i, page in enumerate(doctr_ocr["pages"])]
    }


def get_images(file_bytes, mime_type):
    if mime_type == "application/pdf":
        doc = DocumentFile.from_pdf(file_bytes)
    else:
        doc = DocumentFile.from_images(file_bytes)

    return doc


def ocr_images(images: List[ndarray]):
    result = model(images)

    raw_result = result.export()
    return transform_doctr_ocr(raw_result)
