FROM python:3.10-slim

RUN pip3 install --no-cache-dir --upgrade pip &&  \
    apt-get update &&  \
    apt-get install -y build-essential git software-properties-common &&  \
    apt-get clean


COPY . /home/appuser/speech-to-text-finetune/local_data
COPY . /home/appuser/speech-to-text-finetune/src
COPY . /home/appuser/speech-to-text-finetune/pyproject.toml
WORKDIR /home/appuser/speech-to-text-finetune

RUN pip3 install /home/appuser/speech_to_text_finetune &&  \
    groupadd --gid 1000 appuser &&  \
    useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

USER appuser

ENTRYPOINT ["python", "src/speech_to_text_finetune/finetune_whisper.py"]
