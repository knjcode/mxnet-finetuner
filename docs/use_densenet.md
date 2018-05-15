# Use DenseNet

ImageNet pretrained DenseNet-169 model is introduced on [A MXNet implementation of DenseNet with BC structure].

To use DenseNet-169 pretrained models, specify the `imagenet1k-densenet-169` in `config.yml`.

`mxnet-finetuner` is designed to download the DenseNet-169 model automatically,
but if it is not downloaded automatically, you can use this model as below.

## Download parameter and symbol files

densenet-imagenet-169-0-0125.params  
https://drive.google.com/open?id=0B_M7XF_l0CzXX3V3WXJoUnNKZFE

densenet-imagenet-169-0-symbol.json  
https://raw.githubusercontent.com/bruinxiong/densenet.mxnet/master/densenet-imagenet-169-0-symbol.json

## Rearrange downloaded files

Change the name of the downloaded files and store it as below.

```
model/imagenet1k-densenet-169-0000.params
model/imagenet1k-densenet-169-symbol.json
```


[A MXNet implementation of DenseNet with BC structure]: https://github.com/bruinxiong/densenet.mxnet
