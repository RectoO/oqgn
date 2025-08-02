# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
# ENV DOCTR_MULTIPROCESSING_DISABLE=true
# ENV DOCTR_CACHE_DIR=/tmp
# ENV MPLCONFIGDIR=/tmp
# ENV LAMBDA_TASK_ROOT=/var/task

WORKDIR /var/www


ARG HUGGINGFACE_CACHE_DIR=./huggingface_cache
ENV HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_CACHE_DIR
RUN mkdir $HUGGINGFACE_CACHE_DIR

# Install
RUN apt-get update && \
    apt-get install -y \
    libmagic1 \
    build-essential \
    cmake \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    pkg-config \
    libhdf5-dev \
    poppler-utils ffmpeg libsm6 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt ./

# Upgrade pip
RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ./config.json /var/www/config.json
COPY ./src /var/www/src
RUN mkdir /var/www/models
COPY ./models/ocr /var/www/models/ocr
RUN mkdir /var/www/models/skwiz
RUN mkdir /var/www/tmp

RUN python -m src.skwiz.download_base_layoutlm_models
RUN chmod -R 0777 $HUGGINGFACE_CACHE_DIR

CMD ["python -m", "src.app.py"]
