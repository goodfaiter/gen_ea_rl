# Base python image
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Install and update system dependencies
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install --upgrade pip

# Install torch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install python dependencies
RUN pip3 install openai-harmony transformers kernels accelerate

# Install dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y git 

RUN apt-get update && apt install -y mesa-utils

# Create workspace
RUN mkdir workspace
WORKDIR /workspace