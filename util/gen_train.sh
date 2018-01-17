#!/usr/bin/env bash

# Generate train and validation image record files.
# Settings other than image_size are read from config.yml
#
# Usage:
#   $ util/gen_train.sh <config.yml> <image_size>
#   $ util/gen_train.sh /config/config.yml 224

set -u

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$CUR_DIR/functions"

CONFIG_FILE="$1"
RESIZE="$2"

python3 -c 'import sys, yaml, json; json.dump(yaml.safe_load(sys.stdin), sys.stdout, indent=2)' < "$CONFIG_FILE" > config.json
config=$(jq -Mc '.' config.json)

TRAIN="/images/train"
VALID="/images/valid"
DATA_TRAIN="/data/train"
DATA_VALID="/data/valid"
mkdir -p $DATA_TRAIN $DATA_VALID

NUM_THREAD=$(get_conf "$config" ".common.num_threads" "4")
TRAIN_RATIO=$(get_conf "$config" ".data.train_ratio" "1")
QUALITY=$(get_conf "$config" ".data.quality" "95")
SHUFFLE=$(get_conf "$config" ".data.shuffle" "1")
CENTER_CROP=$(get_conf "$config" ".data.center_crop" "0")

echo "TRAIN_RATIO=$TRAIN_RATIO"
echo "RESIZE=$RESIZE"

if [[ $CENTER_CROP = 1 ]]; then
  CENTER_CROP="True"
else
  CENTER_CROP="False"
fi
echo "CENTER_CROP=$CENTER_CROP"

if [[ $SHUFFLE = 1 ]]; then
  SHUFFLE="True"
else
  SHUFFLE="False"
fi
echo "SHUFFLE=$SHUFFLE"


# Generate train image list from train directory.
python3 -u /mxnet/tools/im2rec.py --list True --recursive True \
                                 --shuffle "${SHUFFLE}" --train-ratio "${TRAIN_RATIO}" \
                                 "images-train-${RESIZE}" "${TRAIN}/"

if [[ "$TRAIN_RATIO" != "1" ]]; then
  # TRAIN_RATIO < 1.0
  # Generate validation image record file from train directory.
  mv "images-train-${RESIZE}_train.lst" "images-train-${RESIZE}.lst"
  mv "images-train-${RESIZE}_val.lst" "images-valid-${RESIZE}.lst"
  python3 -u /mxnet/tools/im2rec.py --resize "${RESIZE}" --quality "${QUALITY}" --shuffle "${SHUFFLE}" \
                                   --num-thread "${NUM_THREAD}" --center-crop "${CENTER_CROP}" \
                                   "images-valid-${RESIZE}" "${TRAIN}/"
  mv images-valid* "${DATA_VALID}"

  # Create valid labels.txt from train directory
  find ${TRAIN}/* -type d | LC_ALL=C sort | awk -F/ '{print NR-1, $NF}' > ${DATA_VALID}/labels.txt

  # Create valid images.txt from train directory
  LC_ALL=C $CUR_DIR/counter.sh "${TRAIN}" | sed -e '1d' > ${DATA_VALID}/images-valid-${RESIZE}.txt
else
  # TRAIN_RATIO = 1.0
  # Generate validation image list from valid directory.
  python3 -u /mxnet/tools/im2rec.py --list True --recursive True \
                                   --shuffle "${SHUFFLE}" --train-ratio 1.0 \
                                   "images-valid-${RESIZE}" "${VALID}/"

  # Check whether validation images exist.
  VALID_IMAGES_NUM=$(echo $(cat "images-valid-${RESIZE}.lst" | wc -l))
  if [[ "$VALID_IMAGES_NUM" = 0 ]]; then
    echo 'Error: Validation images do not exist.' 1>&2
    echo 'Please put validation images in images/valid direcotory.' 1>&2
    echo 'or' 1>&2
    echo 'Set train_ratio in config.yml smaller than 1.0 to use part of train images for validation.' 1>&2
    exit 1
  else
    # Generate validation image record file.
    python3 -u /mxnet/tools/im2rec.py --resize "${RESIZE}" --quality "${QUALITY}" --shuffle "${SHUFFLE}" \
                                    --num-thread "${NUM_THREAD}" --center-crop "${CENTER_CROP}" \
                                    "images-valid-${RESIZE}" ${VALID}/
    mv images-valid* "${DATA_VALID}"
  fi

  # Create valid labels.txt
  find ${VALID}/* -type d | LC_ALL=C sort | awk -F/ '{print NR-1, $NF}' > ${DATA_VALID}/labels.txt

  # Create valid images.txt
  LC_ALL=C $CUR_DIR/counter.sh "${VALID}" | sed -e '1d' > ${DATA_VALID}/images-valid-${RESIZE}.txt
fi

# Check wheter train images exist.
TRAIN_IMAGES_NUM=$(echo $(cat "images-train-${RESIZE}.lst" | wc -l))
if [[ $TRAIN_IMAGES_NUM = 0 ]]; then
  echo 'Error: Train images do not exist.' 1>&2
  echo 'Please put train images in images/train direcotory.' 1>&2
  exit 1
fi

# Generate train image record file.
python3 -u /mxnet/tools/im2rec.py --resize "${RESIZE}" --quality "${QUALITY}" --shuffle "${SHUFFLE}" \
                                 --num-thread "${NUM_THREAD}" --center-crop "${CENTER_CROP}" \
                                 "images-train-${RESIZE}" ${TRAIN}/
mv images-train* "${DATA_TRAIN}"

# Create train labels.txt
find ${TRAIN}/* -type d | LC_ALL=C sort | awk -F/ '{print NR-1, $NF}' > ${DATA_TRAIN}/labels.txt

# Create train images.txt
LC_ALL=C $CUR_DIR/counter.sh "${TRAIN}" | sed -e '1d' > ${DATA_TRAIN}/images-train-${RESIZE}.txt
