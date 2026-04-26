FROM python:3.11-slim

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
    && python -m pip install .

ENTRYPOINT ["bertnado"]
