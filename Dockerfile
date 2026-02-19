
# FROM python:3.10-slim
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    espeak-ng \
    libsndfile1 \
    build-essential\
    ffmpeg 

# RUN apt-get install -y wget xz-utils && \
#     wget https://www.johnvansickle.com/ffmpeg/old-releases/ffmpeg-6.0.1-amd64-static.tar.xz && \
#     tar xvf ffmpeg-6.0.1-amd64-static.tar.xz && \
#     mv ffmpeg-6.0.1-amd64-static/ffmpeg /usr/local/bin/ && \
#     mv ffmpeg-6.0.1-amd64-static/ffprobe /usr/local/bin/ && \
#     rm -rf ffmpeg-6.0.1-amd64-static* && \
#     rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ==================================================================
# conda install and conda forge channel as default
# ------------------------------------------------------------------
# Install miniforge
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /opt/nvidia/entrypoint.d/100.conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${VENV}" >> /opt/nvidia/entrypoint.d/110.conda_default_env.sh && \
    echo "conda activate ${VENV}" >> $HOME/.bashrc

ENV PATH /opt/conda/bin:$PATH

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict

ARG VENV_NAME="cosyvoice"
ENV VENV=$VENV_NAME

RUN conda create -y -n ${VENV} python=3.10
ENV CONDA_DEFAULT_ENV=${VENV}
ENV PATH /opt/conda/bin:/opt/conda/envs/${VENV}/bin:$PATH

RUN conda activate ${VENV} && conda install -y -c conda-forge pynini==2.1.5

COPY backend/CosyVoice/requirements.txt ./backend/CosyVoice/requirements.txt
RUN conda activate ${VENV} && \
    pip install "setuptools<81.0.0" && \
    pip install --no-cache-dir -r backend/CosyVoice/requirements.txt
COPY backend/requirements.txt ./backend/requirements.txt
RUN conda activate ${VENV} && pip install --no-cache-dir -r backend/requirements.txt


COPY backend ./backend
ENV PORT=8080

ENV PYTHONUNBUFFERED=1

RUN mkdir -p /app/backend/CosyVoice/pretrained_models

WORKDIR /app/backend

RUN chmod +x ./entrypoint.sh

CMD ["./entrypoint.sh"]
