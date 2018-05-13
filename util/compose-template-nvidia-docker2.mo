version: "2.3"

services:
  finetuner:
    image: knjcode/mxnet-finetuner
    runtime: nvidia
    environment:
      - SLACK_API_TOKEN
      - MXNET_CUDNN_AUTOTUNE_DEFAULT
    ports:
      - "8888:8888"
    volumes:
      - "$PWD:/config:ro"
      - "$PWD/images:/images:rw"
      - "$PWD/data:/data:rw"
      - "$PWD/model:/mxnet/example/image-classification/model:rw"
      - "$PWD/logs:/mxnet/example/image-classification/logs:rw"
      - "$PWD/classify_example:/mxnet/example/image-classification/classify_example:rw"

  mms:
    image: awsdeeplearningteam/mms_gpu
    runtime: nvidia
    command: "mxnet-model-server start --mms-config /model/mms_app_gpu.conf"
    ports:
      - "8080:8080"
    volumes:
      - "$PWD/model:/model"
