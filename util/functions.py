#!/usr/bin/env python
# coding: utf-8

# Utility functions

def get_image_size(model_prefix):
  if 'caffenet' in model_prefix:
    return 227
  elif 'squeezenet' in model_prefix:
    return 227
  elif 'alexnet' in model_prefix:
    return 227
  elif 'googlenet' in model_prefix:
    return 299
  elif 'inception-v3' in model_prefix:
    return 299
  elif 'inception-v4' in model_prefix:
    return 299
  elif 'inception-resnet-v2' in model_prefix:
    return 299
  else:
    return 224
