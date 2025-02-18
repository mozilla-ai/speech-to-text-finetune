FROM python:3.10-slim

RUN pip3 install --no-cache-dir --upgrade pip &&  \
    apt-get update &&  \
    apt-get install -y build-essential git software-properties-common && \
    apt-get clean

COPY . /home/appuser/speech-to-text-finetune
WORKDIR /home/appuser/speech-to-text-finetune

USER root
RUN pip3 install -e . && \
    groupadd --gid 1000 appuser &&  \
    useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

USER appuser

ENTRYPOINT ["bash"]
