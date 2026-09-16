FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    git wget curl \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    python -m ensurepip --upgrade && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python -m pip install --no-cache-dir \
    torch>=2.7.0 torchvision>=0.22.0 --index-url https://download.pytorch.org/whl/cu128

RUN python -m pip install --no-cache-dir "lerobot[aloha,pusht]"

RUN python -m pip install --no-cache-dir \
    h5py>=3.10.0 \
    pyarrow>=14.0.0 \
    pandas>=2.1.0 \
    matplotlib>=3.8.0 \
    seaborn>=0.13.0 \
    scipy>=1.11.0 \
    wandb>=0.16.0 \
    tqdm>=4.66.0 \
    imageio[ffmpeg]>=2.33.0 \
    rosbags>=0.9.0 \
    mcap>=1.1.0 \
    ruff>=0.4.0

WORKDIR /workspace
COPY . /workspace/

ENV HF_HOME=/data/huggingface
ENV WANDB_DIR=/workspace/wandb

CMD ["bash"]
