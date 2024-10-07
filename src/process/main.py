import magic

from src.ocr.main import get_images, ocr_images
from src.process.post_process import process_classification_page
from src.process.process_ara import process_ara
from src.process.process_b62 import process_b62
from src.process.process_bp import process_bp
from src.process.process_bpk import process_bpk
from src.process.process_del import process_del
from src.process.process_lng import process_lng
from src.process.process_occidental import process_occidental
from src.process.process_ooc import process_ooc
from src.skwiz.models import classify_page

accepted_mime_types = [
    "application/pdf",
    "image/jpeg",
    "image/jpg",
    "image/png",
]


def read_file(file_path: str):
    # Read file
    with open(file_path, "rb") as input_file:
        file_bytes = input_file.read()

    # Check mime type
    mime_type = magic.from_buffer(file_bytes, mime=True)

    return file_bytes, mime_type


def process_file(file_path: str, config):
    file_bytes, mime_type = read_file(file_path)

    if mime_type not in accepted_mime_types:
        raise ValueError(f"Unsupported file type: {mime_type}")

    images = get_images(file_bytes, mime_type)

    # First page classification
    first_page_ocr_result = ocr_images(images[0:1])
    first_page_ocr = first_page_ocr_result["pages"][0]
    classified_page = classify_page(
        model_name="classifier-oqgn",
        image=images[0],
        page_ocr=first_page_ocr,
    )
    classification = process_classification_page(classified_page)

    if classification == "B62":
        response = process_b62(
            images, config["mapping"]["B62"]["fc1"], config["mapping"]["B62"]["fc2"]
        )
    elif classification == "ARA":
        response = process_ara(images, first_page_ocr, config["mapping"]["ARA"])
    elif classification == "BP":
        response = process_bp(images, first_page_ocr, config["mapping"]["BP"])
    elif classification == "BP (Epsilon)":
        response = process_bpk(images, first_page_ocr, config["mapping"]["BPK"])
    elif classification == "Occidental":
        response = process_occidental(images, first_page_ocr, config["mapping"]["OXS"])
    elif classification == "LNG":
        response = process_lng(images, first_page_ocr, config["mapping"]["LNG"])
    elif classification == "DEL":
        response = process_del(images, first_page_ocr, config["mapping"]["DEL"])
    elif classification == "OOC":
        response = process_ooc(
            images,
            first_page_ocr,
            config["mapping"]["OOC"]["stream1"],
            config["mapping"]["OOC"]["stream2"],
        )
    else:
        raise ValueError(f"Unsupported classification: {classification}")

    return response
