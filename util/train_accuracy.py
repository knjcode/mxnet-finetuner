#!/usr/bin/env python
# coding: utf-8

# Create a train accuracy graph
#
# Usage:
#   $ ./train_accuracy.py <config.yml> <output filename> <train logs> ...
#   $ ./train_accuracy.py /config/config.yml logs/$MODEL_PREFIX-$MODEL-$OPTIMIZER.png logs/$MODEL_PREFIX-$MODEL-$OPTIMIZER.log ...

import re
import os
import sys
import yaml
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

config_file = sys.argv[1]
output_file = sys.argv[2]
train_logs = sys.argv[3:]

with open(config_file) as rf:
    config = yaml.safe_load(rf)

try:
    config = config['finetune']
except AttributeError:
    print('Error: Missing finetune section at config.yml')
    sys.exit(1)

tg_fontsize = config.get('train_accuracy_graph_fontsize', 12)
tg_figsize = config.get('train_accuracy_graph_figsize', '8,6')
tg_figsize = tuple(float(i) for i in tg_figsize.split(','))

train_log_filename = os.path.basename(train_logs[0])
model_prefix = os.path.splitext(train_log_filename)[0]

model_regex = re.compile(r'pretrained_model=\'(\S+)\'')
model_regex_scratch = re.compile(r'network=\'(\S+)\'')
lr_regex = re.compile(r'lr=(\S+),')


gpus_regex = re.compile(r'gpus=(\S+),')
load_epoch_regex = re.compile(r'load_epoch=(\S+),')
batch_size_regex = re.compile(r'batch_size=(\d+),')
num_of_image_regex = re.compile(r'num_examples=(\d+),')
optimizer_regex = re.compile(r'optimizer=\'(\S+)\'')
top_k_regex = re.compile(r'top_k=(\S+),')

with open(train_logs[0], 'r') as l:
    data = l.read()
    try:
        model = re.search(model_regex, data).groups()[0]
    except AttributeError:
        model = re.search(model_regex_scratch, data).groups()[0]
    lr = float(re.search(lr_regex, data).groups()[0])
    try:
        load_epoch = int(re.search(load_epoch_regex, data).groups()[0])
    except ValueError:
        load_epoch = 0
    batch_size = re.search(batch_size_regex, data).groups()[0]
    num_of_image = num_of_image_regex.search(data).groups()[0]
    optimizer = optimizer_regex.search(data).groups()[0]
    top_k = int(re.search(top_k_regex, data).groups()[0])

batch_per_epoch = math.ceil(float(num_of_image)/float(batch_size))

# train accuracy per batch
# ex. Epoch[1] Batch [10]	Speed: 20.16 samples/sec	accuracy=0.648148
train_acc_batch = re.compile(r'Epoch\[(\d+)\] Batch \[(\d+)\]\s+Speed:\s+(\S+)\s+samples/sec\s+accuracy=(\S+)')

# train accuracy per epoch
# ex. Epoch[2] Train-accuracy=0.916667
train_acc_epoch = re.compile(r'Epoch\[(\d+)\] Train-accuracy=(\S+)')

# validation accuracy per epoch
# ex. Epoch[3] Validation-accuracy=1.000000
val_acc_epoch = re.compile(r'Epoch\[(\d+)\] Validation-accuracy=(\S+)')

# top-k accuracy per epoch
# ex. Epoch[4] Validation-top_k_accuracy_5=1.000000
top_k_acc_epoch = re.compile(r'Epoch\[(\d+)\] Validation-top_k_accuracy_\d=(\S+)')

train_acc_x = []
train_acc_y = []
train_speed = []
val_acc_x = []
val_acc_y = []
val_acc_x_top_k = []
val_acc_y_top_k = []

for train_log in train_logs:
    with open(train_log, 'r') as l:
        lines = l.readlines()

        for line in lines:
            line = line.strip()

            match = train_acc_batch.search(line)
            if match:
                items = match.groups()
                epoch = int(items[0])
                batch = int(items[1])
                speed = float(items[2])
                acc = float(items[3])
                iteration = epoch * batch_per_epoch + batch
                train_acc_x.append(iteration)
                train_acc_y.append(acc)
                train_speed.append(speed)

            match = train_acc_epoch.search(line)
            if match:
                items = match.groups()
                epoch = int(items[0])
                acc = float(items[1])
                iteration = epoch * batch_per_epoch + batch_per_epoch
                train_acc_x.append(iteration)
                train_acc_y.append(acc)

            match = val_acc_epoch.search(line)
            if match:
                items = match.groups()
                epoch = int(items[0])
                acc = float(items[1])
                iteration = epoch * batch_per_epoch + batch_per_epoch
                val_acc_x.append(iteration)
                val_acc_y.append(acc)

            if top_k > 0:
                match = top_k_acc_epoch.search(line)
                if match:
                    items = match.groups()
                    epoch = int(items[0])
                    acc = float(items[1])
                    iteration = epoch * batch_per_epoch + batch_per_epoch
                    val_acc_x_top_k.append(iteration)
                    val_acc_y_top_k.append(acc)

# print train_acc_x, train_acc_y
# print val_acc_x, val_acc_y
# print val_acc_x_top_k, val_acc_y_top_k

max_acc_index = np.argmax(val_acc_y)
max_acc = val_acc_y[max_acc_index]
if top_k > 0:
    max_top_k_acc_index = np.argmax(val_acc_y_top_k)
    max_top_k_acc = val_acc_y_top_k[max_top_k_acc_index]

sns.set()
fig = plt.figure(figsize = tg_figsize)
plt.rcParams["font.size"] = tg_fontsize

plt.plot(train_acc_x, train_acc_y, "r", label="train accuracy")
plt.plot(val_acc_x, val_acc_y, "b", label="validation accuracy")
if top_k > 0:
    plt.plot(val_acc_x_top_k, val_acc_y_top_k, "g", label="top-" + str(top_k) + " val-acc")

plt_x = train_acc_x[0]
plt.text(plt_x, 0.35, '   model used:    %s' % model)
plt.text(plt_x, 0.30, '   optimizer:       %s' % optimizer)
plt.text(plt_x, 0.25, '   learning rate:  %s' % lr)
plt.text(plt_x, 0.20, '   result val-acc: %s (%s epoch)' % (val_acc_y[-1], len(val_acc_y)+load_epoch))
plt.text(plt_x, 0.15, '   best val-acc:   %s (%s epoch)' % (max_acc, max_acc_index+1+load_epoch))
plt.text(plt_x, 0.10, '   train speed:    %.2f (samples/sec) (batch size: %s)' % (np.mean(train_speed), batch_size))
if top_k > 0:
    plt.text(plt_x, 0.05, '   best top-%s val-acc: %s (%s epoch)' % (top_k, max_top_k_acc, max_top_k_acc_index+1))

plt.title('model accuracy\n%s' % model_prefix)
plt.xlabel("Iterations")
plt.ylabel("Accuracy")

plt.ylim(0,1)
plt.legend(loc=4)

fig.tight_layout()

plt.savefig(output_file)
print("Saved train accuracy graph to \"%s\"" % output_file)
