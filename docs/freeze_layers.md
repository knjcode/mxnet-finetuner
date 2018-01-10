# How to freeze layers during fine-tuning

If you set the number of target layer to `finetune.num_active_layers` in `config.yml` as below, only layers whose number is not greater than the number of the specified layer will be train.

```
finetune:
  models:
    - imagenet1k-nin
  optimizers:
    - sgd
  num_active_layers: 6
```

The default for `finetune.num_active_layers` is 0, in which case all layers are trained.

If you set `1` to `finetune.num_active_layers`, only the last fully-connected layers are trained.

You can check the layer numbers of various pretrained models with `num_layers` command.

```
$ docker-compose run finetuner num_layers <pretrained model name>
```

An example of checking the layer number of `imagenet1k-caffenet` is as follows.

```
$ docker-compose run finetuner num_layers imagenet1k-caffenet
(...snip...)
Number of the layer of imagenet1k-caffenet
   27: data
   26: conv1_weight
   25: conv1_bias
   24: conv1_output
   23: relu1_output
   22: pool1_output
   21: norm1_output
   20: conv2_weight
   19: conv2_bias
   18: conv2_output
   17: relu2_output
   16: pool2_output
   15: norm2_output
   14: conv3_weight
   13: conv3_bias
   12: conv3_output
   11: relu3_output
   10: conv4_weight
    9: conv4_bias
    8: conv4_output
    7: relu4_output
    6: conv5_weight
    5: conv5_bias
    4: conv5_output
    3: relu5_output
    2: pool5_output
    1: flatten_0_output
If you set the number of a layer displayed above to num_active_layers in config.yml,
only layers whose number is not greater than the number of the specified layer will be train.
```
