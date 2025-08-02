import json

from src.constants import CONFIG_FILE, TAG_DEFAULT_VALUES_FILE
from src.supplier_processing.process_ara import process_ara
from src.supplier_processing.process_b62 import process_b62
from src.supplier_processing.process_bpg import process_bpg
from src.supplier_processing.process_bpk import process_bpk
from src.supplier_processing.process_del import process_del
from src.supplier_processing.process_lng import process_lng
from src.supplier_processing.process_ooc import process_ooc
from src.supplier_processing.process_oxs import process_oxs
from src.utils import update_tag_default_values


with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

with open(TAG_DEFAULT_VALUES_FILE, "r", encoding="utf-8") as f:
    tag_default_values = json.load(f)

def supplier_processing(classification, images, first_page_ocr):
      # Process file
    if classification == "B62":
        response = process_b62(
            images, config["mapping"]["B62"]["fc1"], config["mapping"]["B62"]["fc2"]
        )
    elif classification == "ARA":
        response = process_ara(images, first_page_ocr, config["mapping"]["ARA"])
    elif classification == "BPG":
        response = process_bpg(images, first_page_ocr, config["mapping"]["BPG"])
    elif classification == "BPK":
        response = process_bpk(images, first_page_ocr, config["mapping"]["BPK"])
    elif classification == "OXS":
        response = process_oxs(
            images,
            first_page_ocr,
            config["mapping"]["OXS"]["stream1"],
            config["mapping"]["OXS"]["stream2"],
        )
    elif classification == "LNG":
        response = process_lng(images, first_page_ocr, config["mapping"]["LNG"])
    elif classification == "OOC":
        response = process_ooc(
            images,
            first_page_ocr,
            config["mapping"]["OOC"]["stream1"],
            config["mapping"]["OOC"]["stream2"],
        )
    elif classification == "DEL":
        response = process_del(images, first_page_ocr, config["mapping"]["DEL"])
    else:
        raise ValueError(f"Unsupported classification: {classification}")

    (date, extracted_data) = response
    updated_extracted_data = update_tag_default_values(extracted_data, tag_default_values)

    return (date, updated_extracted_data)
