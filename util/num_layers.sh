#!/usr/bin/env bash

set -eu

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$CUR_DIR/functions"

MODEL="$1"
LAYER_BEFORE_FULLC=$(get_layer_before_fullc "$MODEL")
python3 util/fine-tune.py --pretrained-model $MODEL --layer-before-fullc $LAYER_BEFORE_FULLC --num-classes 2 --print-layers-and-exit
