#!/usr/bin/env python
# coding: utf-8

# Predict with specified model and generate predict_results.txt
# Settings other than image_size are read from config.yml
#
# Usage:
#   $ ./predict.py <config.yml> <image_size> <"test" or "valid"> <model> <epoch>
#   $ ./predict.py /config/config.yml 224 test 201705292200-imagenet1k-nin-sgd 3

import os
import cv2
import heapq
import numpy as np
import sys
import yaml
sys.path.append(os.getcwd())
from common import find_mxnet
import mxnet as mx
from collections import namedtuple

config_file = sys.argv[1]
size = int(sys.argv[2])
target = sys.argv[3]
model_prefix = sys.argv[4]
model_epoch = int(sys.argv[5])

with open(config_file) as conf:
    config = yaml.safe_load(conf)

try:
    use_latest = config['test'].get('use_latest', 1)
    batch_size = config['test'].get('test_batch_size', 10)
    top_k = config['test'].get('top_k', 10)
    rgb_mean = config['test'].get('rgb_mean', '123.68,116.779,103.939')
    rgb_mean = [mean for mean in rgb_mean.split(',')]
except AttributeError:
    print('Error: Missing test section at config.yml')
    sys.exit(1)


if top_k < 1:
    print('Error top_k must bigger than 0')
    sys.exit(1)

data_shape = (3,size,size)
try:
    gpus = str(config['common'].get('gpus', ''))
except AttributeError:
    gpus = ''

data_train="/data/train"
data_valid="/data/valid"
data_test="/data/test"
latest_result_log="logs/latest_result.txt"

Batch = namedtuple('Batch', ['data'])

if use_latest:
    with open(latest_result_log) as r:
        model_prefix, model_epoch = r.read().splitlines()
    model_epoch = int(model_epoch)

if target == 'test':
    data_dir = data_test
elif target == 'valid':
    data_dir = data_valid
    # if target is 'valid' use latest fine-tuned model
    # Overwrite model and epoch from latest_result_log
    with open(latest_result_log) as r:
        model_prefix, model_epoch = r.read().splitlines()
    model_epoch = int(model_epoch)
else:
    print('Error: Invalid target name. Please specify `test` or `valid`.')
    sys.exit(1)

print("model_prefix: %s" % model_prefix)
print("model_epoch: %s" % model_epoch)


def load_model(model_prefix, model_epoch, batch_size, size, gpus):
    sym, arg_params, aux_params = mx.model.load_checkpoint('model/' + model_prefix, model_epoch)
    # devices for training
    devs = mx.cpu() if gpus is None or gpus is '' else [mx.gpu(int(i)) for i in gpus.split(',')]
    mod = mx.mod.Module(symbol=sym, context=devs, label_names=['softmax_label'])
    mod.bind(for_training=False,
             data_shapes=[('data', (batch_size,3,size,size))],
             label_shapes=[('softmax_label', (batch_size,))])
    mod.set_params(arg_params, aux_params, allow_missing=True)
    return mod


def load_image_record(imgrec, batch_size, data_shape):
    rec = mx.io.ImageRecordIter(
        path_imgrec         = imgrec,
        label_width         = 1,
        mean_r              = rgb_mean[0],
        mean_g              = rgb_mean[1],
        mean_b              = rgb_mean[2],
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        rand_crop           = False,
        rand_mirror         = False,
        num_parts           = 1,
        part_index          = 0)
    return rec


def make_predict_results(imgrec, batch_size, data_shape, imglst, labels_txt, results_log, top_k, gpus):
    test_rec = load_image_record(imgrec, batch_size, data_shape)

    with open(imglst) as lst:
        test_list = [(l.split('\t')[1].strip(), l.split('\t')[2].strip().replace(' ', '_')) for l in lst.readlines()]

    with open(labels_txt) as syn:
        labels = [l.split(' ')[-1].strip() for l in syn.readlines()]

    with open(results_log, 'w') as result:
        result.write("model_prefix: %s\n" % model_prefix)
        result.write("model_epoch: %s\n" % model_epoch)
        result.write("data: %s\n" % imgrec)

        mod = load_model(model_prefix, model_epoch, batch_size, size, gpus)
        for preds, i_batch, batch in mod.iter_predict(test_rec, reset=False):
            for batch_index, (pred, label) in enumerate(zip(preds[0].asnumpy(), batch.label[0].asnumpy())):
                sorted_pred = heapq.nlargest(top_k, enumerate(pred), key=lambda x: x[1])
                results = []
                for sorted_index, value in sorted_pred:
                    results.append("%s %s" % (sorted_index, value))
                list_index = i_batch * batch_size + batch_index
                result.write("%s %s %s\n" % (test_list[list_index][1], int(float(test_list[list_index][0])), ' '.join(results)))


imgrec =  "%s/images-%s-%d.rec" % (data_dir, target, size)
imglst = "%s/images-%s-%d.lst" % (data_dir, target, size)
labels_txt = "model/%s-labels.txt" % model_prefix
results_log = "logs/%s-%04d-%s-results.txt" % (model_prefix, model_epoch, target)

make_predict_results(imgrec, batch_size, data_shape, imglst, labels_txt, results_log, top_k, gpus)
print("Saved predict results to \"%s\"" % results_log)
