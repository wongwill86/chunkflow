# VERSION 0.1.0
# AUTHOR: Will Wong
# DESCRIPTION: Alpine base image with dockerized chunkflow
# BUILD: docker build --rm -t wongwill86/chunkflow -f /docker/Dockerfile .
# SOURCE: https://github.com/wongwill86/chunkflow

FROM python:3.6-alpine3.7

ENV CHUNKFLOW_USER=chunkflow
ENV CHUNKFLOW_HOME=/usr/local/chunkflow

WORKDIR ${CHUNKFLOW_HOME}/chunkflow
COPY . .
RUN ls ${CHUNKFLOW_HOME}/chunkflow

WORKDIR ${CHUNKFLOW_HOME}
RUN set -ex \
    && apk add --no-cache bash \
    && apk add --no-cache --virtual .build-dependencies \
        build-base \
        python-dev \
        libffi-dev \
        openblas-dev \
        linux-headers \
        jpeg-dev \
        zlib-dev \
        shadow \
    && pip install --upgrade pip \
    && pip install chunkflow/ \
    && groupadd -r ${CHUNKFLOW_USER} \
    && useradd -r -d ${CHUNKFLOW_HOME} -g ${CHUNKFLOW_USER} -s /bin/bash ${CHUNKFLOW_USER} \
    && chown -R ${CHUNKFLOW_USER}: ${CHUNKFLOW_HOME} \
    && find /usr/local \
        \( -type d -a -name test -o -name tests \) \
        -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) \
        -exec rm -rf '{}' + \
    && runDeps="$( \
        scanelf --needed --nobanner --recursive /usr/local \
                | awk '{ gsub(/,/, "\nso:", $2); print "so:" $2 }' \
                | sort -u \
                | xargs -r apk info --installed \
                | sort -u \
    )" \
    && apk add --virtual .run-dependencies $runDeps \
    && apk del .build-dependencies

