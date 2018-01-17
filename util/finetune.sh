#!/usr/bin/env bash

set -u

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$CUR_DIR/functions"

CONFIG_FILE="/config/config.yml"

python3 -c 'import sys, yaml, json; json.dump(yaml.safe_load(sys.stdin), sys.stdout, indent=2)' < $CONFIG_FILE > config.json
config=$(jq -Mc '.' config.json)

TRAIN="/images/train"
VALID="/images/valid"
DATA_TRAIN="/data/train"
DATA_VALID="/data/valid"

DATA_NTHREADS=$(get_conf "$config" ".common.num_threads" "4")
GPUS=$(get_conf "$config" ".common.gpus" "")
if [[ ! $GPUS = "" ]]; then
  GPU_OPTION="--gpus $GPUS"
else
  GPU_OPTION=""
fi
echo "GPU_OPTION=$GPU_OPTION"

MODELS=$(get_conf_array "$config" ".finetune.models" "")
if [[ "$MODELS" = "" ]]; then
  MODELS=$(get_conf_array "$config" ".finetune.pretrained_models" "imagenet1k-nin")
fi
echo "MODELS=$MODELS"
OPTIMIZERS=$(get_conf_array "$config" ".finetune.optimizers" "sgd")
echo "OPTIMIZERS=$OPTIMIZERS"
NUM_EPOCHS=$(get_conf "$config" ".finetune.num_epochs" "10")
LOAD_EPOCH=$(get_conf "$config" ".finetune.load_epoch" "0")
if [[ ! $NUM_EPOCHS -gt $LOAD_EPOCH ]]; then
  echo 'Error: num_epochs must be bigger than load_epoch' 1>&2
  exit 1
fi
if [[ ! $LOAD_EPOCH = "0" ]]; then
  LOAD_EPOCH_OPTION="--load-epoch $LOAD_EPOCH"
else
  LOAD_EPOCH_OPTION=""
fi
echo "LOAD_EPOCH_OPTION=$LOAD_EPOCH_OPTION"
LR=$(get_conf "$config" ".finetune.lr" "0.00001")
LR_FACTOR=$(get_conf "$config" ".finetune.lr_factor" "0.1")
LR_STEP_EPOCHS=$(get_conf "$config" ".finetune.lr_step_epochs" "10")
MOM=$(get_conf "$config" ".finetune.mom" "0.9")
WD=$(get_conf "$config" ".finetune.wd" "0.00001")
BATCH_SIZE=$(get_conf "$config" ".finetune.batch_size" "16")
DISP_BATCHES=$(get_conf "$config" ".finetune.disp_batches" "20")
TOP_K=$(get_conf "$config" ".finetune.top_k" "0")
DATA_AUG_LEVEL=$(get_conf "$config" ".finetune.data_aug_level" "0")
RANDOM_CROP=$(get_conf "$config" ".finetune.random_crop" "0")
RANDOM_MIRROR=$(get_conf "$config" ".finetune.random_mirror" "0")
MAX_RANDOM_H=$(get_conf "$config" ".finetune.max_random_h" "0")
MAX_RANDOM_S=$(get_conf "$config" ".finetune.max_random_s" "0")
MAX_RANDOM_L=$(get_conf "$config" ".finetune.max_random_l" "0")
MAX_RANDOM_ASPECT_RATIO=$(get_conf "$config" ".finetune.max_random_aspect_ratio" "0")
MAX_RANDOM_ROTATE_ANGLE=$(get_conf "$config" ".finetune.max_random_rotate_angle" "0")
MAX_RANDOM_SHEAR_RATIO=$(get_conf "$config" ".finetune.max_random_shear_ratio" "0")
MAX_RANDOM_SCALE=$(get_conf "$config" ".finetune.max_random_scale" "1")
MIN_RANDOM_SCALE=$(get_conf "$config" ".finetune.min_random_scale" "1")
RGB_MEAN=$(get_conf "$config" ".finetune.rgb_mean" "123.68,116.779,103.939")
MONITOR=$(get_conf "$config" ".finetune.monitor" "0")
PAD_SIZE=$(get_conf "$config" ".finetune.pad_size" "0")
NUM_ACTIVE_LAYERS=$(get_conf "$config" ".finetune.num_active_layers" "0")
AUTO_TEST=$(get_conf "$config" ".finetune.auto_test" "1")
TRAIN_ACCURACY_GRAPH_OUTPUT=$(get_conf "$config" ".finetune.train_accuracy_graph_output" "1")
TRAIN_SLACK_UPLOAD=$(get_conf "$config" ".finetune.train_accuracy_graph_slack_upload" "0")
TRAIN_SLACK_CHANNELS=$(get_conf_array "$config" ".finetune.train_accuracy_graph_slack_channels" "general")

CONFUSION_MATRIX_OUTPUT=$(get_conf "$config" ".test.confusion_matrix_output" "1")
TEST_SLACK_UPLOAD=$(get_conf "$config" ".test.confusion_matrix_slack_upload" "0")
TEST_SLACK_CHANNELS=$(get_conf_array "$config" ".test.confusion_matrix_slack_channels" "general")
CLASSIFICATION_REPORT_OUTPUT=$(get_conf "$config" ".test.classification_report_output" "1")

# data_aug_level
echo "DATA_AUG_LEVEL=$DATA_AUG_LEVEL"
if [[ $DATA_AUG_LEVEL -ge 1 ]]; then
  RANDOM_CROP="1"
  RANDOM_MIRROR="1"
fi
if [[ $DATA_AUG_LEVEL -ge 2 ]]; then
  MAX_RANDOM_H="36"
  MAX_RANDOM_S="50"
  MAX_RANDOM_L="50"
fi
if [[ $DATA_AUG_LEVEL -ge 3 ]]; then
  MAX_RANDOM_ASPECT_RATIO="0.25"
  MAX_RANDOM_ROTATE_ANGLE="10"
  MAX_RANDOM_SHEAR_RATIO="0.1"
fi

for MODEL in $MODELS; do
  # Determine IMAGE_SIZE
  IMAGE_SIZE=$(get_conf "$config" ".data.resize_short" "0")
  MODEL_IMAGE_SIZE=$(get_image_size "$MODEL")
  if [[ $IMAGE_SIZE -eq 0 ]]; then
    IMAGE_SIZE=$MODEL_IMAGE_SIZE
  fi
  if [[ $IMAGE_SIZE -lt $MODEL_IMAGE_SIZE ]]; then
    echo 'Error: The shorter edge after resizing must be grater than or equal to the input size of the model.' 1>&2
    echo 'Check data.resize_short at config.yml.' 1>&2
    echo "When using $MODEL model, data.resize_short must be $MODEL_IMAGE_SIZE or more."
    exit 1
  fi
  IMAGE_SHAPE="3,$MODEL_IMAGE_SIZE,$MODEL_IMAGE_SIZE"

  # Construct train commnd
  if [[ $(check_from_scratch "$MODEL") -eq 0 ]]; then
    # from scratch
    echo "Training from scratch $MODEL"
    MODEL=$(trim_scratch "$MODEL")
    NUM_LAYERS=$(get_num_layers "$MODEL")
    if [[ "$NUM_LAYERS" = 'null' ]]; then
      # do not have num-layers
      TRAIN_COMMAND="python util/train_imagenet.py --network $MODEL"
    else
      # num-layers
      TRIMED_MODEL=$(trim_num_layers "$MODEL")
      TRAIN_COMMAND="python util/train_imagenet.py --network $TRIMED_MODEL --num-layers $NUM_LAYERS"
    fi
  else
    # fine-tuning
    # Determine LAYER_BEFORE_FULLC and IMAGE_SIZE
    LAYER_BEFORE_FULLC=$(get_layer_before_fullc "$MODEL")
    TRAIN_COMMAND="python util/fine-tune.py --pretrained-model $MODEL --layer-before-fullc $LAYER_BEFORE_FULLC --num-active-layers $NUM_ACTIVE_LAYERS"
  fi
  echo "TRAIN_COMMAND=$TRAIN_COMMAND"

  # If necessary image records do not exist, they are generated.
  if [ "$DATA_TRAIN/images-train-$IMAGE_SIZE.rec" -ot "$TRAIN" ]; then
    echo "$DATA_TRAIN/images-train-$IMAGE_SIZE.rec does not exist or is outdated." 1>&2
    echo 'Generate image records for fine-tuning.' 1>&2
    $CUR_DIR/gen_train.sh "$CONFIG_FILE" "$IMAGE_SIZE" || exit 1
  fi
  if [ "$DATA_VALID/images-valid-$IMAGE_SIZE.rec" -ot "$VALID" ]; then
    echo "$DATA_VALID/images-valid-$IMAGE_SIZE.rec does not exist or is outdated." 1>&2
    echo 'Generate validation image records for fine-tuning.' 1>&2
    $CUR_DIR/gen_train.sh "$CONFIG_FILE" "$IMAGE_SIZE" || exit 1
  fi
  if [ "$DATA_VALID/images-valid-$MODEL_IMAGE_SIZE.rec" -ot "$VALID" ]; then
    echo "$DATA_VALID/images-valid-$MODEL_IMAGE_SIZE.rec does not exist or is outdated." 1>&2
    echo 'Generate validation image records for fine-tuning.' 1>&2
    $CUR_DIR/gen_train.sh "$CONFIG_FILE" "$MODEL_IMAGE_SIZE" || exit 1
  fi

  # Check the number of image files. If it is different from previous one, regenerate images records
  diff --brief <(LC_ALL=C $CUR_DIR/counter.sh $TRAIN | sed -e '1d') <(cat $DATA_TRAIN/images-train-$IMAGE_SIZE.txt) > /dev/null 2>&1
  if [ "$?" -eq 1 ]; then
    echo "$DATA_TRAIN/images-train-$IMAGE_SIZE.rec is outdated." 1>&2
    echo 'Generate image records for fine-tuning.' 1>&2
    $CUR_DIR/gen_train.sh "$CONFIG_FILE" "$IMAGE_SIZE" || exit 1
  fi
  diff --brief <(LC_ALL=C $CUR_DIR/counter.sh $VALID | sed -e '1d') <(cat $DATA_VALID/images-valid-$IMAGE_SIZE.txt) > /dev/null 2>&1
  if [ "$?" -eq 1 ]; then
    echo "$DATA_VALID/images-valid-$IMAGE_SIZE.rec is outdated." 1>&2
    echo 'Generate validation image records for fine-tuning.' 1>&2
    $CUR_DIR/gen_train.sh "$CONFIG_FILE" "$IMAGE_SIZE" || exit 1
  fi
  diff --brief <(LC_ALL=C $CUR_DIR/counter.sh $VALID | sed -e '1d') <(cat $DATA_VALID/images-valid-$MODEL_IMAGE_SIZE.txt) > /dev/null 2>&1
  if [ "$?" -eq 1 ]; then
    echo "$DATA_VALID/images-valid-$MODEL_IMAGE_SIZE.rec is outdated." 1>&2
    echo 'Generate validation image records for fine-tuning.' 1>&2
    $CUR_DIR/gen_train.sh "$CONFIG_FILE" "$MODEL_IMAGE_SIZE" || exit 1
  fi


  LABELS_TRAIN="$DATA_TRAIN/labels.txt"
  LABELS_VALID="$DATA_VALID/labels.txt"
  diff --brief "$LABELS_TRAIN" "$LABELS_VALID" > /dev/null
  if [ "$?" -eq 1 ]; then
    echo 'Error: The directory structure of images/train and images/valid is different.' 1>&2
    echo 'Check your train and validation images.' 1>&2
    exit 1
  fi

  NUM_CLASSES=$(echo $(cat "$DATA_TRAIN/labels.txt" | wc -l))
  NUM_EXAMPLES=$(echo $(cat "$DATA_TRAIN/images-train-$IMAGE_SIZE.lst" | wc -l))

  for OPTIMIZER in $OPTIMIZERS; do
    MODEL_PREFIX="$(date +%Y%m%d%H%M%S)-$MODEL-$OPTIMIZER"
    LOGS="logs/$MODEL_PREFIX.log"

    # copy labels.txt
    LABELS="model/$MODEL_PREFIX-labels.txt"
    cp "$DATA_TRAIN/labels.txt" "$LABELS"

    $TRAIN_COMMAND \
    --data-train "$DATA_TRAIN/images-train-${IMAGE_SIZE}.rec" \
    --data-val "$DATA_VALID/images-valid-${IMAGE_SIZE}.rec" \
    $GPU_OPTION \
    --num-epochs "$NUM_EPOCHS" \
    $LOAD_EPOCH_OPTION \
    --lr "$LR" \
    --lr-factor "$LR_FACTOR" \
    --lr-step-epochs "$LR_STEP_EPOCHS" \
    --optimizer "$OPTIMIZER" \
    --mom "$MOM" --wd "$WD" \
    --batch-size "$BATCH_SIZE" \
    --disp-batches "$DISP_BATCHES" \
    --top-k "$TOP_K" \
    --data-nthreads "$DATA_NTHREADS" \
    --random-crop "$RANDOM_CROP" \
    --random-mirror "$RANDOM_MIRROR" \
    --max-random-h "$MAX_RANDOM_H" \
    --max-random-s "$MAX_RANDOM_S" \
    --max-random-l "$MAX_RANDOM_L" \
    --max-random-aspect-ratio "$MAX_RANDOM_ASPECT_RATIO" \
    --max-random-rotate-angle "$MAX_RANDOM_ROTATE_ANGLE" \
    --max-random-shear-ratio "$MAX_RANDOM_SHEAR_RATIO" \
    --max-random-scale "$MAX_RANDOM_SCALE" \
    --min-random-scale "$MIN_RANDOM_SCALE" \
    --rgb-mean "$RGB_MEAN" \
    --monitor "$MONITOR" \
    --pad-size "$PAD_SIZE" \
    --image-shape "$IMAGE_SHAPE" \
    --num-classes "$NUM_CLASSES" \
    --num-examples "$NUM_EXAMPLES" \
    --model-prefix "model/$MODEL_PREFIX" 2>&1 | tee "$LOGS"

    if [ "${PIPESTATUS[0]}" -eq 0 ]; then
      # Record model_prefix and best validation accuracy epoch
      echo "$MODEL_PREFIX" > logs/latest_result.txt
      COUNT=$(grep 'Validation-acc' "logs/$MODEL_PREFIX.log" | sort -t'=' -k2 | tail -n 1 | cut -d'[' -f2 | cut -d']' -f1)
      MODEL_EPOCH=$((COUNT + 1))
      echo "$MODEL_EPOCH" >> logs/latest_result.txt

      if [[ $TRAIN_ACCURACY_GRAPH_OUTPUT = 1 ]]; then
        IMAGE="logs/$MODEL_PREFIX-train_accuracy.png"
        python3 util/train_accuracy.py "$CONFIG_FILE" "$IMAGE" "$LOGS"
        if [[ $TRAIN_SLACK_UPLOAD = 1 ]]; then
          python3 util/slack_file_upload.py "$TRAIN_SLACK_CHANNELS" "$IMAGE"
        fi
      fi

      if [[ $AUTO_TEST = 1 ]]; then
        echo 'Start auto test using fine-tuned model with validation data'
        LABELS="model/$MODEL_PREFIX-labels.txt"

        python3 util/predict.py "$CONFIG_FILE" "$MODEL_IMAGE_SIZE" "valid" "$MODEL_PREFIX" "$MODEL_EPOCH"

        # Make a confusion matrix from prediction results.
        if [[ $CONFUSION_MATRIX_OUTPUT = 1 ]]; then
          PREDICT_RESULTS_LOG="logs/$MODEL_PREFIX-epoch$MODEL_EPOCH-valid-results.txt"
          IMAGE="logs/$MODEL_PREFIX-epoch$MODEL_EPOCH-valid-confusion_matrix.png"
          python3 util/confusion_matrix.py "$CONFIG_FILE" "$LABELS" "$IMAGE" "$PREDICT_RESULTS_LOG"

          if [[ $TEST_SLACK_UPLOAD = 1 ]]; then
            python3 util/slack_file_upload.py "$TEST_SLACK_CHANNELS" "$IMAGE"
          fi
        fi
        # Make a classification report from prediction results.
        if [[ $CLASSIFICATION_REPORT_OUTPUT = 1 ]]; then
          PREDICT_RESULTS_LOG="logs/$MODEL_PREFIX-epoch$MODEL_EPOCH-valid-results.txt"
          REPORT="logs/$MODEL_PREFIX-epoch$MODEL_EPOCH-valid-classification_report.txt"
          python3 util/classification_report.py "$CONFIG_FILE" "$LABELS" "$PREDICT_RESULTS_LOG" "$REPORT"
          if [[ -e "$REPORT" ]]; then
            print_classification_report "$REPORT"
          else
            echo 'Error: classification report does not exist.' 1>&2
          fi
        fi
      fi
    else
      echo "Error(Exit code ${PIPESTATUS[0]}): Failed to fine-tune: $MODEL_PREFIX" 1>&2
    fi

  done
done
