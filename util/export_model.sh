#!/usr/bin/env bash

# Generate MXNet models for mxnet-model-server

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$CUR_DIR/functions"
source "$CUR_DIR/vendor/mo"

CONFIG_FILE="/config/config.yml"

python3 -c 'import sys, yaml, json; json.dump(yaml.safe_load(sys.stdin), sys.stdout, indent=2)' < $CONFIG_FILE > config.json
config=$(jq -Mc '.' config.json)

DATA_TRAIN="/data/train"
LATEST_RESULT_LOG="logs/latest_result.txt"

USE_LATEST=$(get_conf "$config"  ".export.use_latest" "1")
TOP_K=$(get_conf "$config"  ".export.top_k" "10")
RGB_MEAN=$(get_conf "$config" ".export.rgb_mean" "123.68,116.779,103.939")
CENTER_CROP=$(get_conf "$config"  ".export.center_crop" "1")

MODEL_NAME=$(get_conf "$config" ".export.model_name" "model")

if [[ $USE_LATEST = 1 ]]; then
  # Check latest_result.txt
  MODEL=$(head -n 1 $LATEST_RESULT_LOG)
  EPOCH=$(tail -n 1 $LATEST_RESULT_LOG)
else
  MODEL_AND_EPOCH=$(get_conf "$config"  ".export.model" "")
  if [[ "$MODEL_AND_EPOCH" = "" ]]; then
    echo 'Error: export.model in config.yml is empty.' 1>&2
    exit 1
  fi
  # Get model_prefix and epoch
  MODEL=${MODEL_AND_EPOCH%-*}
  EPOCH=$(echo $MODEL_AND_EPOCH|rev|cut -d'-' -f1|rev|sed "s/0*\([0-9]*[0-9]$\)/\1/g")
fi

PARAMS="model/$MODEL-$(printf '%04d' $EPOCH).params"
SYMBOL_JSON="model/$MODEL-symbol.json"
LABELS_TXT="model/$MODEL-labels.txt"

# Check existence of .params file
if [ ! -e $PARAMS ]; then
  echo "Error: $PARAMS does not exist." 1>&2
  exit 1
elif [ ! -e $SYMBOL_JSON ]; then
  echo "Error: $SYMBOL_JSON does not exist." 1>&2
  exit 1
elif [ ! -e $LABELS_TXT ]; then
  echo "Error: $LABELS_TXT does not exist." 1>&2
  exit 1
fi

echo "Start generating $MODEL.model" 1>&2

NUM_CLASSES=$(echo $(cat "model/$MODEL-labels.txt" | wc -l))
if [ $TOP_K -gt $NUM_CLASSES ]; then
  TOP_K=$NUM_CLASSES
  echo "INFO: TOP_K must less or equal NUM_CLASSES. Set TOP_K=$NUM_CLASSES" 1>&2
fi
# Determine MODEL_IMAGE_SIZE
MODEL_IMAGE_SIZE=$(get_image_size "$MODEL")

export_tmp_dir=$(mktemp -d tmp.XXXXXXXXXX)
service_tmp_dir=$(mktemp -d tmp.XXXXXXXXXX)

cp $PARAMS $export_tmp_dir \
&& echo "Use $PARAMS" 1>&2

cp $SYMBOL_JSON $export_tmp_dir \
&& echo "Use $SYMBOL_JSON" 1>&2

cat $LABELS_TXT | cut -d' ' -f2 > $export_tmp_dir/synset.txt \
&& echo "Use $LABELS_TXT as synset.txt" 1>&2

# save config.yml
CONFIG_LOG="logs/$MODEL-$(printf "%04d" $EPOCH)-export-config.yml"
cp "$CONFIG_FILE" "$CONFIG_LOG" \
&& echo "Saved config file to \"$CONFIG_LOG\"" 1>&2

generate_export_model_signature "$CUR_DIR" "$MODEL_IMAGE_SIZE" "$RGB_MEAN" "$NUM_CLASSES" "$export_tmp_dir"
generate_export_model_service "$CUR_DIR" "$CENTER_CROP" "$TOP_K" "$service_tmp_dir"
generate_export_model_conf "$CUR_DIR" "$MODEL_NAME" "$MODEL.model"

mxnet-model-export --model-name "$MODEL" --model-path "$export_tmp_dir" --service "$service_tmp_dir/mxnet_finetuner_service.py" \
&& cp $MODEL.model model/ \
&& echo "Saved model to \"model/$MODEL.model\"" 1>&2
