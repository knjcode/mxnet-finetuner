#!/usr/bin/env bash

ln /dev/null /dev/raw1394

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$CUR_DIR/util/functions"

TASK="$1"
ARG1="$2"

CONFIG_FILE="/config/config.yml"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: config.yml does not exist."
  exit 1
fi

if [[ "$TASK" = 'finetune' ]]; then
  util/finetune.sh
elif [[ "$TASK" = 'gen_train' ]]; then
  if [[ "$ARG1" = '' ]]; then
    IMAGE_SIZE=224
  else
    IMAGE_SIZE="$ARG1"
  fi
  util/gen_train.sh "$CONFIG_FILE" "$IMAGE_SIZE"
elif [[ "$TASK" = 'gen_test' ]]; then
  if [[ "$ARG1" = '' ]]; then
    IMAGE_SIZE=224
  else
    IMAGE_SIZE="$ARG1"
  fi
  util/gen_test.sh "$CONFIG_FILE" "$IMAGE_SIZE"
elif [[ "$TASK" = 'test' ]]; then
  util/test.sh
elif [[ "$TASK" = 'num_layers' ]]; then
  util/num_layers.sh "$ARG1"
elif [[ "$TASK" = 'jupyter' ]]; then
  jupyter notebook --no-browser --port=8888 --ip=0.0.0.0 --allow-root
elif [[ "$TASK" = 'version' ]]; then
  version
elif [[ "$TASK" = 'help' ]]; then
  usage
else
  if [[ "$TASK" = '' ]]; then
    util/finetune.sh
  else
    echo "mxnet-finetuner: '$TASK' is not a mxnet-finetuner command."
    usage
  fi
fi
