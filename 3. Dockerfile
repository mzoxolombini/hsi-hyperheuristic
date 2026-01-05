FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_VISIBLE_DEVICES=0,1,2,3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    libgdal-dev \
    libopencv-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    g++ \
    gcc \
    make \
    cmake \
    git \
    wget \
    curl \
    unzip \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR /workspace

# Copy project files
COPY requirements.txt .
COPY setup.py .
COPY README.md .
COPY config.json .
COPY run.py .
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir -e .

# Create data directories
RUN mkdir -p /workspace/data /workspace/results /workspace/cache /workspace/logs

# Set permissions
RUN chmod +x /workspace/scripts/*.sh

# Download datasets
RUN /workspace/scripts/download_datasets.sh

# Expose ports (for TensorBoard, etc.)
EXPOSE 6006 8888

# Default command
CMD ["python3", "run.py", "--help"]
