# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Modified from https://github.com/dmlc/mxnet/blob/master/example/image-classification/fine-tune.py

import os
import sys
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
sys.path.append(os.getcwd())
from common import find_mxnet
from common import data, fit, modelzoo
import mxnet as mx

def get_fine_tune_model(symbol, arg_params, num_classes, layer_name):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc' not in k})
    return (net, new_args)


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="fine-tune a dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train = fit.add_fit_args(parser)
    data.add_data_args(parser)
    aug = data.add_data_aug_args(parser)
    parser.add_argument('--pretrained-model', type=str,
                        help='the pre-trained model')
    parser.add_argument('--layer-before-fullc', type=str, default='flatten0',
                        help='the name of the layer before the last fullc layer')
    parser.add_argument('--num-active-layers', type=int, default=0,
                        help='num of last N-layers to train. if 0 specified, train all layers')
    # use less augmentations for fine-tune
    # data.set_data_aug_level(parser, 1)
    # use a small learning rate and less regularizations
    parser.set_defaults(image_shape='3,224,224', num_epochs=30,
                        lr=.01, lr_step_epochs='20')

    args = parser.parse_args()

    is_user_model = False
    # load pretrained model
    dir_path = os.path.dirname(os.path.realpath(__file__))
    (prefix, epoch) = modelzoo.download_model(
        # args.pretrained_model, os.path.join(dir_path, 'model'))
        args.pretrained_model, os.path.join(dir_path, '../model'))

    # load user fine-tuned model
    if prefix is None:
        is_user_model = True
        (prefix, epoch) = (os.path.join(dir_path, '../model', args.pretrained_model), args.load_epoch)
        # load_epoch='15' lr_step_epoch='10,20,30' -> lr_step_epoch='25,35,45'
        args.lr_step_epochs = ','.join(map(str, [int(args.load_epoch) + int(ep) for ep in args.lr_step_epochs.split(',')]))
    else:
        args.load_epoch = 0

    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    if is_user_model:
        # not remove the last fullc layer
        (new_sym, new_args) = (sym, arg_params)
    else:
        # remove the last fullc layer
        (new_sym, new_args) = get_fine_tune_model(
            sym, arg_params, args.num_classes, args.layer_before_fullc)

    # freeze layers
    fixed_params = []
    train_params = []
    if args.num_active_layers > 0:
        print('--------------------------------------')
        active_layer_num = args.num_active_layers + 4 # add the last fully-connected layers
        all_layers = new_sym.get_internals()

        freeze_layers = all_layers.list_outputs()[0:-active_layer_num-1]
        active_layers = all_layers.list_outputs()[-active_layer_num-1:-1]

        if len(freeze_layers) > 15:
            print('...(snip)...')
            for layer in freeze_layers[-15:]:
                print(layer)
        else:
            for layer in freeze_layers:
                print(layer)

        print('----- train the following layers -----')

        if len(active_layers) > 15:
            for layer in active_layers[:15]:
                print(layer)
            print('...(snip)...')
        else:
            for layer in active_layers:
                print(layer)

        for k in new_args:
            is_active = False
            for a in active_layers:
                if k == a:
                    is_active = True
                    train_params.append(k)
            if not is_active:
                fixed_params.append(k)

        print('--------------------------------------')
        print("Train the last fc layers and the following layers: %s" % ', '.join(train_params))
    else:
        print("Train all the layers")

    # train
    fit.fit(args        = args,
            network     = new_sym,
            data_loader = data.get_rec_iter,
            arg_params  = new_args,
            aux_params  = aux_params,
            fixed_params_names = fixed_params)
