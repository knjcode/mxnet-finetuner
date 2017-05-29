#!/usr/bin/env bash

# Move the specified number of jpeg images from the target directory to the output directory
# while maintaining the directory structure.
# If there is no output direcotry, it will be created automatically.
#
# Usage:
#   $ util/move_images.sh <num_of_images> <target_dir> <output_dir>
#   $ util/move_images.sh 20 /images/trian /images/valid

set -u
shopt -s expand_aliases

usage_exit() {
  echo 'Error: Wrong number of arguments' 1>&2
  echo 'Usage: util/move_images.sh <num_of_images> <target_dir> <output_dir>' 1>&2
  exit 1
}

if [[ ! "$#" = 3 ]]; then
  usage_exit
fi

IMG_NUM="$1"
TARGET_DIR="$2"
OUTPUT_DIR="$3"

if which shuf > /dev/null; then
  alias shuffle='shuf'
else
  if which gshuf > /dev/null; then
    alias shuffle='gshuf'
  else
    echo 'shuf or gshuf command not found.' 1>&2
    exit 1
  fi
fi

mkdir -p "$OUTPUT_DIR"
for i in "$TARGET_DIR"/*; do
  c=$(basename "$i")
  echo "processing $c"
  mkdir -p "$OUTPUT_DIR/$c"
  for j in $(find "$i" -name '*.jpg' | shuffle | head -n "$IMG_NUM"); do
    mv "$j" "$OUTPUT_DIR/$c/"
  done
done
