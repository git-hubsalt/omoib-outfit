FROM 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:2.2.0-gpu-py310-cu118-ubuntu20.04-sagemaker

ENV SAGEMAKER_MODEL_SERVER_WORKERS=1
ENV SAGEMAKER_MODEL_SERVER_TIMEOUT=600

ENV CACHE_DIR=/tmp/.sagemaker
ENV TS_CACHE_DIR=${CACHE_DIR}
ENV TORCH_HOME=${CACHE_DIR}
ENV TRANSFORMERS_CACHE=${CACHE_DIR}
ENV MODEL_STORE=${CACHE_DIR}/model-store

RUN rm -rf /var/lib/apt/lists/*
RUN apt-get clean && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p ${CACHE_DIR} ${MODEL_STORE} && \
chmod -R 777 ${CACHE_DIR}

COPY requirements.txt /opt/ml
RUN pip install --upgrade pip
RUN pip install -r /opt/ml/requirements.txt

ENV SAGEMAKER_PROGRAM=inference.py