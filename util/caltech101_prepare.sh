#!/usr/bin/env bash

# This file download the caltech 101 dataset
# (http://www.vision.caltech.edu/Image_Datasets/Caltech101/), and split it into
# example_images directory.

set -u
shopt -s expand_aliases

PWD=$(pwd)

# number of images per class for training
TRAIN_NUM=60
VALID_NUM=20
TEST_NUM=20

# target classes (10 classes)
CLASSES="airplanes Motorbikes Faces watch Leopards bonsai car_side ketch chandelier hawksbill"

if [ ! -e "$PWD/101_ObjectCategories.tar.gz" ]; then
  if which wget > /dev/null 2>&1; then
    wget -O "$PWD/101_ObjectCategories.tar.gz" http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
  else
    if which curl > /dev/null 2>&1; then
      curl http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz > "$PWD/101_ObjectCategories.tar.gz"
    else
      echo 'wget and curl commands not found.' 1>&2
      exit 1
    fi
  fi
fi

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

# check example_images
if [ -e "$PWD/example_images" ]; then
  echo './example_images directory already exits. Remove it and retry.' 1>&2
  exit 1
fi

# split into train, validation and test set
tar -xf "$PWD/101_ObjectCategories.tar.gz"
TRAIN_DIR="$PWD/example_images/train"
VALID_DIR="$PWD/example_images/valid"
TEST_DIR="$PWD/example_images/test"
mkdir -p "$TRAIN_DIR" "$VALID_DIR" "$TEST_DIR"
for i in ${PWD}/101_ObjectCategories/*; do
  c=$(basename "$i")
  if echo "$CLASSES" | grep -q "$c"
  then
    echo "processing $c"
    mkdir -p "$TRAIN_DIR/$c" "$VALID_DIR/$c" "$TEST_DIR/$c"
    for j in $(find "$i" -name '*.jpg' | shuffle | head -n "$TRAIN_NUM"); do
      mv "$j" "$TRAIN_DIR/$c/"
    done
    for j in $(find "$i" -name '*.jpg' | shuffle | head -n "$VALID_NUM"); do
      mv "$j" "$VALID_DIR/$c/"
    done
    for j in $(find "$i" -name '*.jpg' | shuffle | head -n "$TEST_NUM"); do
      mv "$j" "$TEST_DIR/$c/"
    done
  fi
done

# touch .gitignore
touch "$TRAIN_DIR/.gitkeep" "$VALID_DIR/.gitkeep" "$TEST_DIR/.gitkeep"

# clean
rm -rf "$PWD/101_ObjectCategories/"
