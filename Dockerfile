# Use the official Ubuntu server image as a base
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# Set non-interactive frontend to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install essential packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install g++ for C++ compilation
RUN apt-get update && apt-get install -y g++

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Install CUDA 12.4
# First add NVIDIA's package repository and public keys
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-4

# Install PyTorch with CUDA 12.4, numpy, and kagglehub
# Using pip for PyTorch ensures compatibility with CUDA version
RUN pip3 install numpy kagglehub

# Set CUDA environment variables
ENV PATH="/usr/local/cuda-12.4/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}"

# Set default command to python3
CMD ["python3"]
