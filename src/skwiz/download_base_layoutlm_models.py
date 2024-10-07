from transformers import AutoProcessor  # type: ignore[import-untyped]
from src.skwiz.constants import AUTO_PROCESSOR_LOCAL_PATH

model = AutoProcessor.from_pretrained(
    "microsoft/layoutlmv3-base", apply_ocr=False
).save_pretrained(AUTO_PROCESSOR_LOCAL_PATH)
