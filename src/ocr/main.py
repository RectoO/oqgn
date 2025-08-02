from typing import List, Tuple
from numpy import ndarray
import numpy as np
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

        all_confidences = [bounding_box["confidence"] for bounding_box in page_bounding_boxes]
        mean_confidence = sum(all_confidences) / len(all_confidences) if len(all_confidences) > 0 else 0
        return {
            "width": width,
            "height": height,
            "boundingBoxes": page_bounding_boxes,
            "confidence": mean_confidence,
        }

    return {
        "pages": [process_page(page, i) for i, page in enumerate(doctr_ocr["pages"])]
    }

def rotate_image(image: ndarray, angle: int):
    if angle == 0:
        return image
    elif angle == 90:
        return np.rot90(image, k=3)  # 90° clockwise = 270° counter-clockwise
    elif angle == 180:
        return np.rot90(image, k=2)
    elif angle == 270:
        return np.rot90(image, k=1)
    else:
        raise ValueError("Angle must be one of 0, 90, 180, or 270 degrees")

def ocr_image(image: ndarray) -> PageInfo:
    result = model([image])
    raw_result = result.export()
    ocr_result = transform_doctr_ocr(raw_result)
    return ocr_result["pages"][0]

def detect_orientation(image: ndarray, page_ocr: PageInfo) -> Tuple[PageInfo, int]:
    page_ocr_0 = page_ocr
    page_ocr_90 = ocr_image(rotate_image(image, 90))
    page_ocr_180 = ocr_image(rotate_image(image, 180))
    page_ocr_270 = ocr_image(rotate_image(image, 270))

    c0 = page_ocr_0["confidence"]
    c90 = page_ocr_90["confidence"]
    c180 = page_ocr_180["confidence"]
    c270 = page_ocr_270["confidence"]

    if c0 > c90 and c0 > c180 and c0 > c270:
        return page_ocr_0, 0
    elif c90 > c0 and c90 > c180 and c90 > c270:
        return page_ocr_90, 90
    elif c180 > c0 and c180 > c90 and c180 > c270:
        return page_ocr_180, 180
    elif c270 > c0 and c270 > c90 and c270 > c180:
        return page_ocr_270, 270

    return page_ocr, 0


def get_images(file_bytes, mime_type):
    if mime_type == "application/pdf":
        doc = DocumentFile.from_pdf(file_bytes)
    else:
        doc = DocumentFile.from_images(file_bytes)

    return doc


def ocr_images(images: List[ndarray]) -> Ocr:
    result = model(images)

    raw_result = result.export()
    return transform_doctr_ocr(raw_result)
