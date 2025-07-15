# Start from Ubuntu 22.04 (Jammy Jellyfish)
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install pip3
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install --upgrade pip

# Install torch GPU
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install ROS2
RUN apt-get update && apt-get install -y software-properties-common curl && \
    add-apt-repository universe && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt-get update && apt-get install -y ros-humble-xacro

# Install dependencies
RUN apt-get update && apt-get install -y git

# Install Python dependencies
RUN pip3 install numpy

# Create workspace
RUN mkdir workspace
WORKDIR /workspace
