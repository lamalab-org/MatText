# Use the CUDA base image
FROM nvidia/cuda:12.0.0-devel-ubuntu20.04

# Install necessary packages
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip

# ARG to pass the GitHub PAT during build
ARG GITHUB_PAT

# Set the working directory
WORKDIR /app/structllm

RUN git clone --branch  container https://$GITHUB_PAT@github.com/lamalab-org/structllm.git /app/structllm
RUN ls -l

# Set the working directory
WORKDIR /app/structllm

RUN pip3 install .
RUN pip3 install git+https://$GITHUB_PAT@github.com/lamalab-org/xtal2txt.git@structllm_dep


ENTRYPOINT ["/bin/bash"]
