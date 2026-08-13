FROM debian:bookworm-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libeigen3-dev \
        python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Makefile main.cpp ./
COPY include ./include
COPY src ./src
COPY exps ./exps

RUN make -j2

USER 10001:10001
CMD ["/app/main"]
