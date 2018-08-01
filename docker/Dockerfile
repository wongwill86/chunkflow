# VERSION 0.1.0
# AUTHOR: Will Wong
# DESCRIPTION: Slim base image with dockerized chunkflow
# BUILD: docker build --rm -t wongwill86/chunkflow -f /docker/Dockerfile .
# SOURCE: https://github.com/wongwill86/chunkflow

FROM python:3.6-slim
LABEL maintainer=wongwill86

ARG GOSU_VERSION=1.10
ARG CHUNKFLOW_USER=chunkflow
ARG CHUNKFLOW_HOME=/usr/local/chunkflow

ENV CHUNKFLOW_USER ${CHUNKFLOW_USER}
ENV CHUNKFLOW_HOME ${CHUNKFLOW_HOME}

WORKDIR ${CHUNKFLOW_HOME}/chunkflow
COPY MANIFEST.in .
COPY CHANGELOG.rst .
COPY LICENSE .
COPY README.rst .
COPY setup.py .
COPY docker/entrypoint.sh docker/entrypoint.sh
COPY src/ src/

# Will drop to chunkflow user from gosu in entrypoint script
USER root

WORKDIR ${CHUNKFLOW_HOME}
RUN apt-get update \
    && buildDeps=' \
        libffi-dev \
        build-essential \
        libblas-dev \
        liblapack-dev \
        libpq-dev \
        software-properties-common \
        wget \
    ' \
    && apt-get install $buildDeps git vim sudo apt-transport-https -y \
    && dpkgArch="$(dpkg --print-architecture | awk -F- '{ print $NF }')" \
    && wget -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$dpkgArch" \
    && pip install chunkflow/ \
    && pip uninstall cloud-volume -y \
    && pip install git+https://github.com/seung-lab/cloud-volume.git@no_threads#egg=cloud-volume \
    && chmod +x /usr/local/bin/gosu \
    && gosu nobody true \
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

ENTRYPOINT ["chunkflow/docker/entrypoint.sh"]
