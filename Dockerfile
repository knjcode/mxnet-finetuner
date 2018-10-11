FROM nvidia/cuda:9.0-cudnn7-devel

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

RUN git clone https://github.com/apache/incubator-mxnet.git mxnet --branch 1.3.0 \
  && cd mxnet \
  && git config user.email "knjcode@gmail.com" \
  && git cherry-pick ceabcaac77543d99246415b2fb2d8c973a830453

# install mxnet-model-server
RUN git clone https://github.com/awslabs/mxnet-model-server.git --branch v0.4.0 \
  && cd mxnet-model-server \
  && pip3 install -e .

RUN pip3 uninstall -y mxnet
RUN pip3 install mxnet-cu90==1.2.1.post1

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

WORKDIR /mxnet/example/image-classification

COPY common /mxnet/example/image-classification/common/
COPY util /mxnet/example/image-classification/util/
COPY docker-entrypoint.sh .

ENTRYPOINT ["./docker-entrypoint.sh"]
