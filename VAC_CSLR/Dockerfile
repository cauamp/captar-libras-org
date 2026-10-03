FROM ubuntu:20.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt /requirements.txt

RUN apt-get update && \
    apt-get install --no-install-recommends -y python3 python3-dev python3-pip g++ ffmpeg git

RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install wheel && \
    git clone --recursive https://github.com/parlance/ctcdecode.git && \
    cd ctcdecode && \
    pip install .

WORKDIR /

RUN rm requirements.txt
