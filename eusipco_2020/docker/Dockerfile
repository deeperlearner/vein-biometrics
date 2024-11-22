# Use a base image with minimal dependencies (Debian in this case)
FROM ubuntu:20.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.7 \
    python3-dev \
    python3-pip \
    ffmpeg libsm6 libxext6 \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip wheel setuptools

# Install Project Specific Dependencies
ADD requirements.txt .
RUN python3 -m pip install -r requirements.txt 
