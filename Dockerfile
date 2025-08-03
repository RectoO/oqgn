# -------- STAGE 1: Build image --------
FROM python:3.10-slim AS builder

WORKDIR /build

# Install system deps needed only to build and install Python packages
# Combine commands and clean up in one RUN instruction for smaller layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    pkg-config \
    libhdf5-dev \
    libmagic1 \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy requirements.txt early to leverage Docker cache
COPY requirements.txt .

RUN pip install --upgrade pip

# Crucial: Explicitly specify CPU-only PyTorch wheels using --extra-index-url
# Also, ensure --no-cache-dir is present
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy app source code and config, but defer model copying
COPY ./src ./src
COPY ./config.json .
# Don't copy models/ocr here yet if they are not needed for initial pip install.
# They will be copied in the final stage.

# -------- STAGE 2: Minimal runtime image --------
FROM python:3.10-slim

WORKDIR /var/www

# Runtime dependencies only (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    libopenblas-dev \
    libhdf5-dev \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy installed Python packages from builder
# Use /usr/local/lib/python3.10/site-packages for specific site-packages
# This is typically where pip installs to with --prefix=/install on Debian-based slim images
COPY --from=builder /install/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
# Also copy any binaries/scripts installed by pip (e.g., in /install/bin)
COPY --from=builder /install/bin /usr/local/bin

# Copy application code and models
COPY --from=builder /build/config.json ./config.json
COPY --from=builder /build/src ./src
COPY ./models/ocr ./models/ocr

# Hugging Face models & app-specific models
ENV HUGGINGFACE_HUB_CACHE=./huggingface_cache
RUN mkdir -p $HUGGINGFACE_HUB_CACHE ./models/skwiz ./tmp

# Pre-download models or perform setup
# This is a major area for bloat.
# Consider if you can download these models outside the Docker build
# and then COPY them in, or if they MUST be downloaded during build.
# If they must be downloaded, ensure they are downloaded to the cache
# and then you can use `chmod` on that cache.
RUN python -m src.skwiz.download_base_layoutlm_models
RUN chmod -R 0777 ./huggingface_cache # Only if really necessary, consider more restrictive permissions

# Consider removing temporary files generated during the model download
# if src.skwiz.download_base_layoutlm_models creates them in the layer.
# This would require modifying that script to clean up after itself or
# adding another RUN command here to clean specific paths.
# Example: RUN rm -rf /tmp/download_artifacts

# Set appropriate ownership for security and permissions
# RUN chown -R appuser:appgroup /var/www
# USER appuser

CMD ["python", "-m", "src.app"]
