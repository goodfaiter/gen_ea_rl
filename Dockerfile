# Base python image
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Install and update system dependencies
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install --upgrade pip

# Install huggingface
RUN pip3 install transformers kernels

# Install torch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# # Install Python dependencies
# RUN pip3 install numpy pybind11 pandas scikit-learn matplotlib tqdm roma PyQt6 pyarrow

# # Install dependencies
# ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y git

# Create workspace
RUN mkdir workspace
WORKDIR /workspace