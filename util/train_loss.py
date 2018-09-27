#!/usr/bin/env python
# coding: utf-8

# Create a train loss graph
#
# Usage:
#   $ ./train_loss.py <config.yml> <output filename> <train logs> ...
#   $ ./train_loss.py /config/config.yml logs/$MODEL_PREFIX-$MODEL-$OPTIMIZER.png logs/$MODEL_PREFIX-$MODEL-$OPTIMIZER.log ...

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

tg_fontsize = config.get('train_loss_graph_fontsize', 12)
tg_figsize = config.get('train_loss_graph_figsize', '8,6')
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

# train cross-entropy per batch
# ex. Epoch[1] Batch [200]  Speed: 1693.44 samples/sec      accuracy=0.109375       cross-entropy=3.198293
train_loss_batch = re.compile(r'Epoch\[(\d+)\] Batch \[(\d+)\]\s+Speed:\s+(\S+)\s+samples/sec\s+.*\s+cross-entropy=(\S+)')

# train cross-entropy per epoch
# ex Epoch[7] Train-cross-entropy=1.577511
train_loss_epoch = re.compile(r'Epoch\[(\d+)\] Train-cross-entropy=(\S+)')

# validation cross-entropy per epoch
# ex. Epoch[9] Validation-cross-entropy=1.339509
val_loss_epoch = re.compile(r'Epoch\[(\d+)\] Validation-cross-entropy=(\S+)')

train_loss_x = []
train_loss_y = []
train_speed = []
val_loss_x = []
val_loss_y = []

for train_log in train_logs:
    with open(train_log, 'r') as l:
        lines = l.readlines()

        for line in lines:
            line = line.strip()

            match = train_loss_batch.search(line)
            if match:
                items = match.groups()
                epoch = int(items[0])
                batch = int(items[1])
                speed = float(items[2])
                ce = float(items[3])
                iteration = epoch * batch_per_epoch + batch
                train_loss_x.append(iteration)
                train_loss_y.append(ce)
                train_speed.append(speed)

            match = train_loss_epoch.search(line)
            if match:
                items = match.groups()
                epoch = int(items[0])
                ce = float(items[1])
                iteration = epoch * batch_per_epoch + batch_per_epoch
                train_loss_x.append(iteration)
                train_loss_y.append(ce)

            match = val_loss_epoch.search(line)
            if match:
                items = match.groups()
                epoch = int(items[0])
                ce = float(items[1])
                iteration = epoch * batch_per_epoch + batch_per_epoch
                val_loss_x.append(iteration)
                val_loss_y.append(ce)

min_loss_index = np.argmin(val_loss_y)
min_loss = val_loss_y[min_loss_index]

sns.set()
fig = plt.figure(figsize = tg_figsize)
plt.rcParams["font.size"] = tg_fontsize

plt.plot(train_loss_x, train_loss_y, "r", label="train loss")
plt.plot(val_loss_x, val_loss_y, "b", label="validation loss")

plt_x = train_loss_x[0] + train_loss_x[-1] * 0.1
plt_y = train_loss_y[0]
plt_diff = max(0.05, train_loss_y[0] * 0.05)
plt.text(plt_x, plt_y - plt_diff * 1, '   model used:     %s' % model)
plt.text(plt_x, plt_y - plt_diff * 2, '   optimizer:        %s' % optimizer)
plt.text(plt_x, plt_y - plt_diff * 3, '   learning rate:   %s' % lr)
plt.text(plt_x, plt_y - plt_diff * 4, '   result val-loss: %s (%s epoch)' % (val_loss_y[-1], len(val_loss_y)+load_epoch))
plt.text(plt_x, plt_y - plt_diff * 5, '   best val-loss:   %s (%s epoch)' % (min_loss, min_loss_index+1+load_epoch))
plt.text(plt_x, plt_y - plt_diff * 6, '   train speed:     %.2f (samples/sec) (batch size: %s)' % (np.mean(train_speed), batch_size))

plt.title('model loss\n%s' % model_prefix)
plt.xlabel("Iterations")
plt.ylabel("loss")

plt.ylim(bottom=0.0)
plt.legend(loc=1)

fig.tight_layout()

plt.savefig(output_file)
print("Saved train loss graph to \"%s\"" % output_file)
