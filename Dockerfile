FROM python:3.10-slim

RUN pip3 install --no-cache-dir --upgrade pip && apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git


COPY . /home/appuser/speech-to-text-finetune
WORKDIR /home/appuser/speech-to-text-finetune

RUN pip3 install /home/appuser/speech_to_text_finetune

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

USER appuser

EXPOSE 8501
ENTRYPOINT ["python", "demo/app.py"]
