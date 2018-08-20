FROM nvidia/cuda:9.0-devel

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y \
    build-essential \
    cmake \
    font-manager \
    fonts-ipaexfont \
    git \
    language-pack-ja \
    libatlas-base-dev \
    libcurl4-openssl-dev \
    libgtest-dev \
    libopencv-dev \
    libprotoc-dev \
    protobuf-compiler \
    python-opencv \
    python-dev \
    python-numpy \
    python-tk \
    python3-dev \
    unzip \
    wget \
  && rm -rf /var/lib/apt/lists/*

RUN cd /usr/src/gtest && cmake CMakeLists.txt && make && cp *.a /usr/lib && \
    cd /tmp && wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py

# install libcudnn 7.0.4.31
ENV CUDNN_VERSION 7.0.4.31
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 \
  && rm -rf /var/lib/apt/lists/*

ENV BUILD_OPTS "USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1"
RUN git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet --branch 1.2.1 \
  && cd mxnet \
  && make -j$(nproc) $BUILD_OPTS \
  && rm -r build

RUN wget --quiet https://github.com/stedolan/jq/releases/download/jq-1.5/jq-linux64 \
  && chmod +x jq-linux64 \
  && mv jq-linux64 /usr/bin/jq

RUN pip3 install \
  attrdict \
  awscli \
  jupyter \
  matplotlib \
  nose \
  nose-timer \
  numpy \
  opencv-python \
  pandas \
  pandas_ml \
  Pillow \
  pylint \
  pyyaml \
  requests \
  seaborn \
  sklearn-pandas \
  slackclient \
  tqdm

# install mxnet-model-server
RUN git clone https://github.com/awslabs/mxnet-model-server.git --branch v0.2.0 \
  && cd mxnet-model-server \
  && pip3 install -e .

# install compiled mxnet
RUN cd mxnet/python && pip3 install -e .

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

WORKDIR /mxnet/example/image-classification

COPY common /mxnet/example/image-classification/common/
COPY util /mxnet/example/image-classification/util/
COPY docker-entrypoint.sh .

ENTRYPOINT ["./docker-entrypoint.sh"]
