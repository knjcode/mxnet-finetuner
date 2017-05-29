#!/usr/bin/env python
# coding: utf-8

# Create an image of confusion matrix from prediction results
#
# Usage:
#   $ ./confusion_matrix.py <config.yml> <labels.txt> <output filename> <predict_results.txt>
#   $ ./confusion_matrix.py /config/config.yml logs/$MODEL-PREFIX-lables.txt logs/confusion_matrix.png logs/predict_results.txt
#
# References
#   http://hayataka2049.hatenablog.jp/entry/2016/12/15/222339
#   http://qiita.com/hik0107/items/67ad4cfbc9e84032fc6b
#

import sys
import yaml
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas_ml import ConfusionMatrix
from sklearn.metrics import confusion_matrix

try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except NameError:
    pass

config_file = sys.argv[1]
labels_file = sys.argv[2]
output_file = sys.argv[3]
result_file = sys.argv[4]

with open(config_file) as rf:
    config = yaml.safe_load(rf)

try:
    cm_fontsize = config['test'].get('confusion_matrix_fontsize', 12)
    cm_figsize = config['test'].get('confusion_matrix_figsize', '16,12')
    cm_figsize = tuple(float(i) for i in cm_figsize.split(','))
except AttributeError:
    print('Error: Missing test and/or data section at config.yml')
    sys.exit(1)

try:
    use_japanese_label = config['data'].get('use_japanese_label', 0)
except AttributeError:
    use_japanese_label = 0

with open(labels_file) as sf:
    labels = [l.split(' ')[-1].strip() for l in sf.readlines()]

with open(result_file) as rf:
    lines = rf.readlines()
    model_prefix = lines[0][14:].strip()
    model_epoch = int(lines[1][13:].strip())
    target_data = lines[2]
    results = [(l.split(' ')[0], l.split(' ')[1], l.split(' ')[2]) for l in lines[3:]]

y_true = [labels[int(i[1])] for i in results]
y_pred = [labels[int(i[2])] for i in results]

if use_japanese_label:
    matplotlib.rcParams['font.family'] = 'IPAexGothic'
    sns.set(font=['IPAexGothic'])
else:
    sns.set()

fig = plt.figure(figsize = cm_figsize)
plt.rcParams["font.size"] = cm_fontsize

cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

sns.heatmap(df_cmx, annot=True, fmt='g', cmap='Blues')

plt.title("Confusion matrix\n%s (%d epoch)\n%s" % (model_prefix, model_epoch, target_data))
plt.xlabel("Predict")
plt.ylabel("Actual")

fig.tight_layout()

plt.savefig(output_file)
print("Saved confusion matrix to \"%s\"" % output_file)
