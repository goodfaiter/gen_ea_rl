# Base python image
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Install and update system dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y python3-pip git mesa-utils

# Install basic python dependecies
RUN pip3 install --upgrade pip uv

# Create python env
RUN uv venv --python 3.12 --seed
ENV PATH="/.venv/bin:${PATH}"

# Install torch
RUN uv pip install torch==2.9.0+cu128 torchvision==0.24.0+cu128 torchaudio==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Install python dependencies
RUN uv pip install openai-harmony transformers kernels accelerate

# Install unsloth for fine-tuning
RUN uv pip install "unsloth[cu128-torch290] @ git+https://github.com/unslothai/unsloth.git"

# Add entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]