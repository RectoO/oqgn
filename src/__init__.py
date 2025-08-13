import warnings
from transformers import logging as hf_logging  # type: ignore[import-untyped]

# Suppress specific warnings
warnings.filterwarnings(
    "ignore", message="Initializing zero-element tensors is a no-op"
)
warnings.filterwarnings(
    "ignore",
    message="The `device` argument is deprecated and will be removed in v5 of Transformers.",
)

# Suppress Hugging Face Transformers info/warnings
hf_logging.set_verbosity_error()
