# Use SE-ResNeXt

ImageNet pretrained SE-ResNeXt-50 model is introduced on [SENet.mxnet].

You can use this model as below

## Download parameter and symbol files

se-resnext-imagenet-50-0-0125.params
https://drive.google.com/uc?id=0B_M7XF_l0CzXOHNybXVWLWZteEE

se-resnext-imagenet-50-0-symbol.json
https://raw.githubusercontent.com/bruinxiong/SENet.mxnet/master/se-resnext-imagenet-50-0-symbol.json

## Rearrange downloaded files

Change the name of the downloaded files and store it as below

```
model/imagenet1k-se-resnext-50-0000.params
model/imagenet1k-se-resnext-50-symbol.json
```

## Modify config

To use SE-ResNeXt-50 pretrained models, specify the `imagenet1k-se-resext-50` in `config.yml`.


[SENet.mxnet]: https://github.com/bruinxiong/SENet.mxnet
