FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu20.04

#FROM python:3.7

RUN apt-get -y update

RUN apt-get install -y git \
    software-properties-common \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN add-apt-repository ppa:deadsnakes/ppa && \
	apt update && \
	apt install python3.6 -y && \
	apt install python3-distutils -y && \
	apt install python3.6-dev -y && \
	apt install build-essential -y && \
	#apt-get install python3-pip -y 8& \
	apt update && apt install -y libsm6 libxext6 ffmpeg && \
	apt-get install -y libxrender-dev
RUN apt-get install python3-pip -y
COPY . /ITCar-PHOLOTINO


RUN python3 -m pip install --upgrade pip
RUN cd ITCar-PHOLOTINO
WORKDIR /ITCar-PHOLOTINO
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

RUN python3 -m pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html -U
RUN python3 -m pip install -r requirements.txt


CMD python3 test_client.py

#ARG IMAGE_NAME
#FROM ${IMAGE_NAME}:11.8.0-runtime-ubuntu20.04 as base
#
#FROM base as base-amd64
#
#ENV NV_CUDNN_VERSION 8.6.0.163
#ENV NV_CUDNN_PACKAGE_NAME "libcudnn8"
#
#ENV NV_CUDNN_PACKAGE "libcudnn8=$NV_CUDNN_VERSION-1+cuda11.8"
#
#FROM base as base-arm64
#
#ENV NV_CUDNN_VERSION 8.6.0.163
#ENV NV_CUDNN_PACKAGE_NAME "libcudnn8"
#
#ENV NV_CUDNN_PACKAGE "libcudnn8=$NV_CUDNN_VERSION-1+cuda11.8"
#
#FROM base-${TARGETARCH}
#
#ARG TARGETARCH
#
#LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
#LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"
#
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    ${NV_CUDNN_PACKAGE} \
#    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
#    && rm -rf /var/lib/apt/lists/*

