
FROM python:3.10-slim


RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    espeak-ng \
    libsndfile1 \
    build-essential

RUN apt-get install -y wget xz-utils && \
    wget https://www.johnvansickle.com/ffmpeg/old-releases/ffmpeg-6.0.1-amd64-static.tar.xz && \
    tar xvf ffmpeg-6.0.1-amd64-static.tar.xz && \
    mv ffmpeg-6.0.1-amd64-static/ffmpeg /usr/local/bin/ && \
    mv ffmpeg-6.0.1-amd64-static/ffprobe /usr/local/bin/ && \
    rm -rf ffmpeg-6.0.1-amd64-static* && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY backend/CosyVoice/requirements.txt ./backend/CosyVoice/requirements.txt
RUN pip install --no-cache-dir -r backend/CosyVoice/requirements.txt
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt


COPY backend ./backend
ENV PORT=8080

ENV PYTHONUNBUFFERED=1

RUN mkdir -p /app/backend/CosyVoice/pretrained_models

WORKDIR /app/backend

CMD ["uvicorn", "FastAPIdemo:app", "--host", "0.0.0.0", "--port", "8080"]
