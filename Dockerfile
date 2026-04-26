FROM python:3.12-slim

ARG PACKAGE_VERSION=0.0.0

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    coreutils \
    g++ \
    gcc \
    git \
    libbz2-dev \
    libcurl4-openssl-dev \
    liblzma-dev \
    libtool \
    make \
    perl \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN python -m pip install --upgrade pip \
    && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BERTNADO="${PACKAGE_VERSION}" \
        python -m pip install .

ENTRYPOINT ["bertnado"]
