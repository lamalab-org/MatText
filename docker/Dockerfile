# Use a Python Slim base image
FROM python:3.9-slim AS base


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        gcc \
        python3-dev \
        python3-pip

# Set the working directory
WORKDIR /app/structllm

# Clone the repository using HTTPS URL with PAT authentication
ARG GITHUB_PAT
RUN git clone --branch container https://$GITHUB_PAT@github.com/lamalab-org/structllm.git /app/structllm && \
    cd /app/structllm && \
    git checkout container


WORKDIR /app/structllm

RUN pip3 install git+https://$GITHUB_PAT@github.com/lamalab-org/xtal2txt.git@structllm_dep

# Set bash as the entry point
ENTRYPOINT ["/bin/bash"]
