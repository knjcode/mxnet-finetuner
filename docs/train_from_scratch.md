# Training from scratch

Edit `config.yml` as below.

```
finetune:
  models:
    - scratch-alexnet
```

You can also run fine-tuning and training from scratch together.

```
finetune:
  models:
    - imagenet1k-inception-v3
    - scratch-inception-v3
```

## Available models traininig from scratch

|from scratch model name        |
|:------------------------------|
|scratch-alexnet                |
|scratch-googlenet              |
|scratch-inception-bn           |
|scratch-inception-resnet-v2    |
|scratch-inception-v3           |
|scratch-inception-v4           |
|scratch-lenet                  |
|scratch-mlp                    |
|scratch-mobilenet              |
|scratch-resnet-N               |
|scratch-resnext-N              |
|scratch-vgg-N                  |

Specify the number of layers for N in scratch-resnet, scratch-resnext and scratch-vgg.

For scratch-resnet and scrach-resnext, N can be set to 18, 34, 50, 101, 152, 200 and 269,
and for scratch-vgg, N can be set to 11, 13, 16 and 19.
