# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

WORKDIR /app

RUN chmod 1777 /tmp
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        libgl1 libglx-mesa0 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf python3.10 /usr/bin/python && \
    ln -sf pip3 /usr/bin/pip

COPY requirements.txt .
# Cài PyTorch với CUDA 12.8 trước — cache lại
RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache \
    pip install "torch>=2.0.1" --index-url https://download.pytorch.org/whl/cu128
RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache \
    pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]