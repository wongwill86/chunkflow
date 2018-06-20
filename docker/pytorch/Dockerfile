# VERSION 0.1.0
# AUTHOR: Will Wong
# DESCRIPTION: Slim base image with dockerized chunkflow and pytorch support
# BUILD: docker build --rm -t wongwill86/chunkflow -f /docker/Dockerfile .
# SOURCE: https://github.com/wongwill86/chunkflow

FROM python:3.6-slim
LABEL maintainer=wongwill86

ENV CHUNKFLOW_USER=chunkflow
ENV CHUNKFLOW_HOME=/usr/local/chunkflow

WORKDIR ${CHUNKFLOW_HOME}/chunkflow
COPY CHANGELOG.rst .
COPY LICENSE .
COPY README.rst .
COPY setup.py .
COPY src/ src/
RUN ls ${CHUNKFLOW_HOME}/chunkflow

WORKDIR ${CHUNKFLOW_HOME}
RUN apt-get update \
    && buildDeps=' \
        libffi-dev \
        build-essential \
        libblas-dev \
        liblapack-dev \
        libpq-dev \
        software-properties-common \
        curl \
    ' \
    && apt-get install $buildDeps git vim sudo apt-transport-https -y \
    && pip install chunkflow/ \
    && pip install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-linux_x86_64.whl \
    && groupadd -r ${CHUNKFLOW_USER} \
    && useradd -r -d ${CHUNKFLOW_HOME} -g ${CHUNKFLOW_USER} -s /bin/bash ${CHUNKFLOW_USER} \
    && chown -R ${CHUNKFLOW_USER}: ${CHUNKFLOW_HOME} \
    && apt-get remove --purge -yqq $buildDeps \
    && apt-get clean -y \
    && apt-get autoremove -y \
    && rm -rf \
        /var/lib/apt/lists/* \
        /tmp/* \
        /var/tmp/* \
        /usr/share/man \
        /usr/share/doc \
        /usr/share/doc-base

# Get commit hash and tags these three files need to be excluded from .dockerignore:
#.git/refs/heads/
#.git/refs/tags/
#.git/HEAD
COPY .git .git
RUN mkdir version
RUN cat .git/refs/$(cat .git/HEAD | sed -e's/ref: refs\///g') > version/COMMIT \
    && grep $(cat version/COMMIT) .git/refs/tags/* -l | xargs -n 1 -r basename > version/TAGS \
    && mv version/COMMIT version/base.COMMIT.slim \
    && mv version/TAGS version/base.TAGS.slim

USER ${CHUNKFLOW_USER}
