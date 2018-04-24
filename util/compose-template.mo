version: "3"

services:
  finetuner:
    image: knjcode/mxnet-finetuner
    environment:
      - SLACK_API_TOKEN
      - MXNET_CUDNN_AUTOTUNE_DEFAULT
    ports:
      - "8888:8888"
    volumes:
      {{#VOLUMES}}
      - "{{.}}"
      {{/VOLUMES}}
      - "$PWD:/config:ro"
      - "$PWD/images:/images:rw"
      - "$PWD/data:/data:rw"
      - "$PWD/model:/mxnet/example/image-classification/model:rw"
      - "$PWD/logs:/mxnet/example/image-classification/logs:rw"
      - "$PWD/classify_example:/mxnet/example/image-classification/classify_example:rw"
    {{#ExistDEV}}
    devices:
      {{#DEVICES}}
      - "{{.}}"
      {{/DEVICES}}
    {{/ExistDEV}}
{{#VOLUMES}}
volumes:
  {{.}}FIX_VOLUME_NAME
    external: true
{{/VOLUMES}}
