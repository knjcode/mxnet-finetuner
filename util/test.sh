#!/usr/bin/env bash

set -u

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$CUR_DIR/functions"

CONFIG_FILE="/config/config.yml"

python3 -c 'import sys, yaml, json; json.dump(yaml.safe_load(sys.stdin), sys.stdout, indent=2)' < $CONFIG_FILE > config.json
config=$(jq -Mc '.' config.json)

TEST="/images/test"
DATA_TEST="/data/test"
LATEST_RESULT_LOG="logs/latest_result.txt"

CONFUSION_MATRIX_OUTPUT=$(get_conf "$config"  ".test.confusion_matrix_output" "1")
SLACK_UPLOAD=$(get_conf "$config"  ".test.confusion_matrix_slack_upload" "0")
SLACK_CHANNELS=$(get_conf_array "$config"  ".test.confusion_matrix_slack_channels" "general")
CLASSIFICATION_REPORT_OUTPUT=$(get_conf "$config"  ".test.classification_report_output" "1")

USE_LATEST=$(get_conf "$config"  ".test.use_latest" "1")
EPOCH_UP_TO=$(get_conf "$config" ".test.model_epoch_up_to" "")

if [[ $USE_LATEST = 1 ]]; then
  # Check latest_result.txt
  MODEL=$(head -n 1 $LATEST_RESULT_LOG)
  EPOCH=$(tail -n 1 $LATEST_RESULT_LOG)
else
  MODEL=$(get_conf "$config"  ".test.model_prefix" "")
  if [[ "$MODEL" = "" ]]; then
    echo 'Error: test.model_prefix in config.yml is empty.' 1>&2
    exit 1
  fi
  EPOCH=$(get_conf "$config"  ".test.model_epoch" "")
  if [[ "$EPOCH" = "" ]]; then
    echo 'Error: test.model_epoch in config.yml is empty.' 1>&2
    exit 1
  fi
fi

# Determine MODEL_IMAGE_SIZE
MODEL_IMAGE_SIZE=$(get_image_size "$MODEL")

# If necessary image records do not exist, they are generated.
if [ "$DATA_TEST/images-test-$MODEL_IMAGE_SIZE.rec" -ot "$TEST" ]; then
  echo "$DATA_TEST/images-test-$MODEL_IMAGE_SIZE.rec does not exist or is outdated." 1>&2
  echo "Generate image records for test." 1>&2
  $CUR_DIR/gen_test.sh "$CONFIG_FILE" "$MODEL_IMAGE_SIZE"
fi

# Check the number of image files. If it is different from previous one, regenerate images records
diff --brief <(LC_ALL=C $CUR_DIR/counter.sh $TEST | sed -e '1d') <(cat $DATA_TEST/images-test-$MODEL_IMAGE_SIZE.txt) > /dev/null 2>&1
if [ "$?" -eq 1 ]; then
  echo "$DATA_TEST/images-test-$MODEL_IMAGE_SIZE.rec is outdated." 1>&2
  echo 'Generate image records for test.' 1>&2
  $CUR_DIR/gen_test.sh "$CONFIG_FILE" "$MODEL_IMAGE_SIZE" || exit 1
fi

# TARGET EPOCHS
if [[ "$EPOCH_UP_TO" ]]; then
  EPOCHS=$(seq $EPOCH $EPOCH_UP_TO)
else
  EPOCHS="$EPOCH"
fi

for CUR_EPOCH in $EPOCHS; do
  # Predict with specified model.
  python3 util/predict.py "$CONFIG_FILE" "$MODEL_IMAGE_SIZE" "test" "$MODEL" "$CUR_EPOCH"

  LABELS="model/$MODEL-labels.txt"
  LABELS_TEST="$DATA_TEST/labels.txt"

  diff --brief "$LABELS" "$LABELS_TEST"
  if [[ "$?" -eq 1 ]]; then
    echo 'The directory structure of images/train and images/test is different.' 1>&2
    echo 'Skip making a confusion matrix and/or a classification report.' 1>&2
  else
    # Make a confusion matrix from prediction results.
    if [[ "$CONFUSION_MATRIX_OUTPUT" = 1 ]]; then
      PREDICT_RESULTS_LOG="logs/$MODEL-epoch$CUR_EPOCH-test-results.txt"
      IMAGE="logs/$MODEL-epoch$CUR_EPOCH-test-confusion_matrix.png"
      python3 util/confusion_matrix.py "$CONFIG_FILE" "$LABELS" "$IMAGE" "$PREDICT_RESULTS_LOG"
      if [[ "$SLACK_UPLOAD" = 1 ]]; then
        python3 util/slack_file_upload.py "$SLACK_CHANNELS" "$IMAGE"
      fi
    fi
    # Make a classification report from prediction results.
    if [[ "$CLASSIFICATION_REPORT_OUTPUT" = 1 ]]; then
      PREDICT_RESULTS_LOG="logs/$MODEL-epoch$CUR_EPOCH-test-results.txt"
      REPORT="logs/$MODEL-epoch$CUR_EPOCH-test-classification_report.txt"
      python3 util/classification_report.py "$CONFIG_FILE" "$LABELS" "$PREDICT_RESULTS_LOG" "$REPORT"
      if [[ -e "$REPORT" ]]; then
        print_classification_report "$REPORT"
      else
        echo 'Error: classification report does not exist.' 1>&2
      fi
    fi
  fi

done
