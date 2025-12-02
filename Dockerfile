
FROM python:3.10-slim


RUN apt-get update && apt-get install -y \
    ffmpeg<7 \
    git \
    git-lfs \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


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
