FROM python:3.10-slim

RUN pip3 install --no-cache-dir --upgrade pip &&  \
    apt-get update &&  \
    apt-get install -y build-essential git software-properties-common && \
    apt-get clean

RUN mkdir -p /home/appuser/speech-to-text-finetune
COPY transcribe_app.py /home/appuser/speech-to-text-finetune
WORKDIR /home/appuser/speech-to-text-finetune

USER root
RUN pip3 install git+https://github.com/mozilla-ai/speech-to-text-finetune.git

EXPOSE 7860
ENTRYPOINT ["python", "transcribe_app.py"]
