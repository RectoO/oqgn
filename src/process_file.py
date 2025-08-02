import os
import shutil
import magic

from src.supplier_processing.main import supplier_processing
from src.constants import PROCESSED_INPUT_FOLDER
from src.ocr.main import detect_orientation, get_images, ocr_images, rotate_image
from src.utils import csv_output_tmp_move, process_classification_page
from src.skwiz_models import classify_page


accepted_mime_types = [
    "application/pdf",
    "image/jpeg",
    "image/jpg",
    "image/png",
]

def process_file(file_path: str, file_name: str):
    # Read file
    file_bytes, mime_type = read_file(file_path)
    if mime_type not in accepted_mime_types:
        raise ValueError(f"Unsupported file type: {mime_type}")

    # Analyse file (Classification + Orientation Detection)
    classification, images, first_page_ocr = analyse_file(file_bytes, mime_type)

    # Supplier processing
    (date, extracted_data) = supplier_processing(classification, images, first_page_ocr)

    # Save response
    csv_output_tmp_move(file_name, classification, date, extracted_data)

    # DEBUG ONLY
    # shutil.copy(file_path, os.path.join(PROCESSED_INPUT_FOLDER, file_name))

    # Clean up
    date_year = date.split('-')[0]
    client_folder = f"{classification}_{date_year}"
    formatted_date = "".join(date.split('-'))
    processed_file_name = f"{classification}_{formatted_date}_{file_name}"
    processed_input_path = os.path.join(
        PROCESSED_INPUT_FOLDER, client_folder, processed_file_name
    )
    # Create output folder if it doesn't exist
    os.makedirs(os.path.dirname(processed_input_path), exist_ok=True)
    shutil.move(file_path, processed_input_path)


def read_file(file_path: str):
    # Read file
    with open(file_path, "rb") as input_file:
        file_bytes = input_file.read()

    # Check mime type
    mime_type = magic.from_buffer(file_bytes, mime=True)

    return file_bytes, mime_type


def analyse_file(file_bytes: bytes, mime_type: str):
    images = get_images(file_bytes, mime_type)

    orientation = 0
    first_page_ocr_result = ocr_images(images[0:1])
    first_page_ocr = first_page_ocr_result["pages"][0]
    first_page_confidence = first_page_ocr["confidence"]
    if first_page_confidence < 0.9:
        (first_page_ocr, orientation) = detect_orientation(images[0], first_page_ocr)
        images = [rotate_image(image, orientation) for image in images]

    classified_page = classify_page(
        model_name="classifier-oqgn",
        image=images[0],
        page_ocr=first_page_ocr,
    )
    classification = process_classification_page(classified_page) if classified_page is not None else None

    return classification, images, first_page_ocr
