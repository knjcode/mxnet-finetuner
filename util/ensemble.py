#!/usr/bin/env python
# coding: utf-8

# Predict with ensemble of specified models and generate predict_results.txt
#
# Usage:
#   $ ./ensemble.py <config.yml> <"test" or "valid"> <model_prefix>
#   $ ./ensemble.py /config/config.yml test ensemble

import os
import re
import cv2
import heapq
import numpy as np
import sys
import yaml
sys.path.append(os.getcwd())
from common import find_mxnet
import mxnet as mx
from collections import namedtuple
import functions

config_file = sys.argv[1]
target = sys.argv[2]
model_prefix = sys.argv[3]

with open(config_file) as conf:
    config = yaml.safe_load(conf)

models = config['ensemble'].get('models')

try:
    weights = config['ensemble'].get('weights', False)
    if weights:
        weights = [float(weight) for weight in weights.split(',')]
    batch_size = config['ensemble'].get('ensemble_batch_size', 10)
    top_k = config['ensemble'].get('top_k', 10)
    rgb_mean = config['ensemble'].get('rgb_mean', '123.68,116.779,103.939')
    rgb_mean = [float(mean) for mean in rgb_mean.split(',')]
    rgb_std = config['ensemble'].get('rgb_std', '1,1,1')
    rgb_std = [float(std) for std in rgb_std.split(',')]
except AttributeError:
    print('Error: Missing ensemble section at config.yml')
    sys.exit(1)


if top_k < 1:
    print('Error top_k must bigger than 0')
    sys.exit(1)

try:
    gpus = str(config['common'].get('gpus', ''))
except AttributeError:
    gpus = ''

data_train="/data/train"
data_valid="/data/valid"
data_test="/data/test"
latest_result_log="logs/latest_result.txt"

if target == 'test':
    data_dir = data_test
elif target == 'valid':
    data_dir = data_valid
else:
    print('Error: Invalid target name. Please specify `test` or `valid`.')
    sys.exit(1)


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
        std_r               = rgb_std[0],
        std_g               = rgb_std[1],
        std_b               = rgb_std[2],
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        rand_crop           = False,
        rand_mirror         = False,
        num_parts           = 1,
        part_index          = 0)
    return rec


def make_ensemble_predict_results(data_dir, target, batch_size, imglst, labels_txt, results_log, top_k, gpus):
    with open(imglst) as lst:
        test_list = [(l.split('\t')[1].strip(), l.split('\t')[2].strip()) for l in lst.readlines()]

    with open(labels_txt) as syn:
        labels = [l.split(' ')[-1].strip() for l in syn.readlines()]

    model_size_array = []
    imgrec_array = []
    for model_prefix in model_prefix_array:
        size = functions.get_image_size(model_prefix)
        model_size_array.append(size)
        imgrec_array.append("%s/images-%s-%d.rec" % (data_dir, target, size))
    print(model_size_array)
    print(imgrec_array)

    with open(results_log, 'w') as result:
        result.write("model_prefix: %s\n" % ','.join(model_prefix_array))
        result.write("model_epoch: %s\n" % ','.join([str(epoch) for epoch in model_epoch_array]))
        result.write("data: %s\n" % ','.join(imgrec_array))

        pred_arrays = []
        for model_prefix, model_epoch, model_size in zip(model_prefix_array, model_epoch_array, model_size_array):

            data_shape = (3,model_size,model_size)
            imgrec = "%s/images-%s-%d.rec" % (data_dir, target, model_size)

            mod = load_model(model_prefix, model_epoch, batch_size, model_size, gpus)
            test_rec = load_image_record(imgrec, batch_size, data_shape)

            pred_array = []
            for preds, i_batch, batch in mod.iter_predict(test_rec, reset=False):
                for batch_index, (pred, label) in enumerate(zip(preds[0].asnumpy(), batch.label[0].asnumpy())):
                    pred_array.append(pred)
            pred_arrays.append(pred_array)
            del mod
            del test_rec

        if weights:
            w = np.array(weights)
            w = w / np.sum(w)
            try:
                preds = np.average(pred_arrays, axis=0, weights=w)
            except ValueError:
                print('Length of weights not compatible with number of models.')
                sys.exit(1)
        else:
            preds = np.average(pred_arrays, axis=0)

        for i in range(len(preds)):
            sorted_pred = heapq.nlargest(top_k, enumerate(preds[i]), key=lambda x: x[1])
            results = []
            for sorted_index, value in sorted_pred:
                results.append("%s %s" % (sorted_index, value))
            result.write("%s %s %s\n" % (test_list[i][1], int(float(test_list[i][0])), ' '.join(results)))


model_prefix_array = [re.sub(r'-\d+$','',model) for model in models]
print(model_prefix_array)
model_epoch_array = [int(re.sub(r'^[\w\-\.]+-', '', model)) for model in models]
print(model_epoch_array)

labels_txt = "model/%s-labels.txt" % model_prefix_array[0]  # use first labels.txt in models
size = functions.get_image_size(model_prefix_array[0])
imglst = "%s/images-%s-%d.lst" % (data_dir, target, size)
results_log = "logs/%s-%s-results.txt" % (model_prefix, target)

make_ensemble_predict_results(data_dir, target, batch_size, imglst, labels_txt, results_log, top_k, gpus)
print("Saved predict results to \"%s\"" % results_log)
