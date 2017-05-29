#!/usr/bin/env python
# coding: utf-8

# Make classification report from prediction results
#
# Usage:
#   $ ./classification_report.py <config.yml> <labels.txt> <predict_results.txt> <output filename>
#   $ ./classification_report.py /config/config.yml model/$MODEL-labels.txt logs/$MODEL-epoch$EPOCH-test-predict_results.txt logs/$MODEL-epoch$EPOCH-test-classification_report.txt
#

import sys
import yaml
from sklearn.metrics import classification_report

try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except NameError:
    pass

config_file = sys.argv[1]
labels_file = sys.argv[2]
result_file = sys.argv[3]
output_file = sys.argv[4]

with open(config_file) as rf:
    config = yaml.safe_load(rf)

try:
    config = config['test']
except AttributeError:
    print('Error: Missing test section at config.yml')
    sys.exit(1)

cr_digits = config.get('classification_report_digits', 3)

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

digits = int(cr_digits)
report = classification_report(y_true, y_pred, target_names=labels, digits=digits)
with open(output_file, 'w') as f:
    f.write("model_prefix: %s\n" % model_prefix)
    f.write("model_epoch: %s\n" % model_epoch)
    f.write("%s\n" % target_data)
    f.write(report)
print("Saved classification report to \"%s\"" % output_file)
