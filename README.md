# mxnet-finetuner

An all-in-one Deep Learning toolkit for image classification to fine-tuning pretrained models using MXNet.


## Prerequisites

- docker
- docker-compose
- jq
- wget or curl

When using NVIDIA GPUs

- nvidia-docker


## Setup

```
$ git clone https://github.com/knjcode/mxnet-finetuner
$ cd mxnet-finetuner
$ bash setup.sh
```


## Example usage


### 1. Arrange images into their respective directories

A training data directory (`images/train`) and validation data directory (`images/valid`) should containing one subdirectory per image class.

For example, arrange training data and validation data as follows.

```
images/
    train/
        airplanes/
            airplane001.jpg
            airplane002.jpg
            ...
        watch/
            watch001.jpg
            watch002.jpg
            ...
    valid/
        airplanes/
            airplane101.jpg
            airplane102.jpg
            ...
        watch/
            watch101.jpg
            watch102.jpg
            ...
```


### 2. Edit config.yml

Edit `config.yml` as you like.

For example
```
common:
  num_threads: 16
  gpus: 0,1,2,3

data:
  quality: 95
  shuffle: 1
  center_crop: 0

finetune:
  models:
    - imagenet1k-resnet-50
  optimizers:
    - sgd
  num_epochs: 30
  lr: 0.0001
  lr_factor: 0.1
  lr_step_epochs: 30
  mom: 0.9
  wd: 0.00001
  batch_size: 10
```


### 3. Do Fine-tuning

```
$ docker-compose run finetuner
```

mxnet-finetuner will automatically execute the followings according to `config.yml`.

- Create RecordIO data from images
- Download pretrained models
- Replace the last fully-connected layer with a new one that outputs the desired number of classes
- Data augumentaion
- Do Fine-tuning
- Make training accuracy graph
- Make confusion matrix
- Upload training accuracy graph and confusion matrix to Slack

Training accuracy graph and/or confusion matrix are save at `logs/` directory.  
Trained models are save at `model/` directory.

Trained models are saved with the following file name for each epoch.
```
model/201705292200-imagenet1k-nin-sgd-0000.params
```

If you want to upload results to Slack, set `SLACK_API_TOKEN` environment variable and edit `config.yml` as below.
```
finetune:
  train_accuracy_graph_slack_upload: 1
test:
  confusion_matrix_slack_upload: 1
```


### 4. Predict with trained models

Select the trained model and epoch you want to use for testing and edit `config.yml`

If you want to use `model/201705292200-imagenet1k-nin-sgd-0001.params`, edit `config.yml` as blow.

```
test:
  model_prefix: 201705292200-imagenet1k-nin-sgd
  model_epoch: 1
```

When you want to use the latest highest validation accuracy trained model, edit `config.yml` as below.

```
test:
  use_latest: 1
```

If set this option, `model_prefix` and `model_epoch` are ignored.

When you are done, you can predict with the following command

```
$ docker-compose run finetuner test
```

Predict result and classification report and/or confusion matrix are save at `logs/` directory.


## Available pretrained models

Classification accuracy of available pretrained models.
(Datasets are ImageNet1K, ImageNet11K and Place365 Challenge)

*Please note that the following results are calculated with different datasets.*
- ResNet accuracy from [Reproduce ResNet-v2 using MXNet]
- Other accuracy from [MXNet model gallery] and [MXNet - Image Classification - Pre-trained Models]

|model                          |Top-1 Accuracy|Top-5 Accuracy|download size|model size (MXNet)|dataset    |image shape|
|:------------------------------|-------------:|-------------:|------------:|-----------------:|----------:|----------:|
|CaffeNet                       |54.5%         |78.3%         |233MB        |9.3MB             |ImageNet1K |227x227    |
|SqueezeNet                     |55.4%         |78.8%         |4.8MB        |4.8MB             |ImageNet1K |227x227    |
|NIN                            |58.8%         |81.3%         |30MB         |30MB              |ImageNet1K |224x224    |
|ResNet-18                      |69.5%         |89.1%         |45MB         |43MB              |ImageNet1K |224x224    |
|VGG16                          |71.0%         |89.8%         |528MB        |58MB              |ImageNet1K |224x224    |
|VGG19                          |71.0%         |89.8%         |549MB        |78MB              |ImageNet1K |224x224    |
|Inception-BN                   |72.5%         |90.8%         |44MB         |40MB              |ImageNet1K |224x224    |
|ResNet-34                      |72.8%         |91.1%         |84MB         |82MB              |ImageNet1K |224x224    |
|ResNet-50                      |75.6%         |92.8%         |98MB         |90MB              |ImageNet1K |224x224    |
|ResNet-101                     |77.3%         |93.4%         |171MB        |163MB             |ImageNet1K |224x224    |
|ResNet-152                     |77.8%         |93.6%         |231MB        |223MB             |ImageNet1K |224x224    |
|ResNet-200                     |77.9%         |93.8          |248MB        |240MB             |ImageNet1K |224x224    |
|Inception-v3                   |76.9%         |93.3%         |92MB         |84MB              |ImageNet1K |299x299    |
|ResNeXt-50                     |76.9%         |93.3%         |96MB         |89MB              |ImageNet1K |224x224    |
|ResNeXt-101                    |78.3%         |94.1%         |170MB        |162MB             |ImageNet1K |224x224    |
|ResNeXt-101-64x4d              |79.1%         |94.3%         |320MB        |312MB             |ImageNet1K |224x224    |
|ResNet-152 (imagenet11k)       |41.6%         |-             |311MB        |223MB             |ImageNet11K|224x224    |
|ResNet-50 (Place365 Challenge) |31.1%         |-             |181MB        |90MB              |Place365ch |224x224    |
|ResNet-152 (Place365 Challenge)|33.6%         |-             |313MB        |223MB             |Place365ch |224x224    |

- The `download size` is the file size when first downloading pretrained model.
- The `model size` is the file size to be saved after fine-tuning.


To use these pretrained models, specify the following pretrained model name in `config.yml`.

|model                          |pretrained model name            |
|:------------------------------|:--------------------------------|
|CaffeNet                       |imagenet1k-caffenet              |
|SqueezeNet                     |imagenet1k-squeezenet            |
|NIN                            |imagenet1k-nin                   |
|VGG16                          |imagenet1k-vgg16                 |
|VGG19                          |imagenet1k-vgg19                 |
|Inception-BN                   |imagenet1k-inception-bn          |
|ResNet-18                      |imagenet1k-resnet-18             |
|ResNet-34                      |imagenet1k-resnet-34             |
|ResNet-50                      |imagenet1k-resnet-50             |
|ResNet-101                     |imagenet1k-resnet-101            |
|ResNet-152                     |imagenet1k-resnet-152            |
|ResNet-152 (imagenet11k)       |imagenet11k-resnet-152           |
|ResNet-200                     |imagenet1k-resnet-200            |
|Inception-v3                   |imagenet1k-inception-v3          |
|ResNeXt-50                     |imagenet1k-resnext-50            |
|ResNeXt-101                    |imagenet1k-resnext-101           |
|ResNeXt-101-64x4d              |imagenet1k-resnext-101-64x4d     |
|ResNet-50 (Place365 Challenge) |imagenet11k-place365ch-resnet-50 |
|ResNet-152 (Place365 Challenge)|imagenet11k-place365ch-resnet-152|


## Benchmark


### Speed (images/sec)

- dataset: 8000 samles
- batch size: 10, 20, 30, 40
- Optimizer: SGD
- GPU: Maxwell TITAN X (12GiB Memory)

|model                          |batch size 10|batch size 20|batch size 30|batch size 40|
|:------------------------------|------------:|------------:|------------:|------------:|
|CaffeNet                       |755.64       |1054.47      |1019.24      |1077.63      |
|SqueezeNet                     |458.27       |579.37       |534.68       |549.55       |
|NIN                            |443.88       |516.21       |612.83       |656.14       |
|ResNet-18                      |257.40       |308.30       |331.57       |339.09       |
|ResNet-34                      |149.88       |182.69       |201.75       |207.49       |
|Inception-BN                   |147.60       |183.82       |193.74       |203.86       |
|ResNet-50                      |88.55        |102.44       |109.98       |111.04       |
|Inception-v3                   |67.11        |75.90        |80.67        |82.34        |
|VGG16                          |56.38        |58.01        |59.80        |59.35        |
|ResNet-101                     |53.42        |63.28        |68.14        |68.35        |
|VGG19                          |45.02        |46.88        |48.62        |48.28        |
|ResNet-152                     |37.88        |44.89        |48.48        |48.62        |
|ResNet-200                     |22.58        |25.61        |27.17        |27.32        |
|ResNeXt-50                     |53.30        |64.40        |71.07        |72.96        |
|ResNeXt-101                    |31.76        |39.56        |42.90        |43.99        |
|ResNeXt-101-64x4d              |18.22        |23.08        |out of memory|out of memory|


### Memory usage (MiB)

- dataset: 8000 samples
- batch size: 10, 20, 30, 40
- Optimizer: SGD
- GPU: Maxwell TITAN X (12GiB GPU Memory)

|model            |batch size 10|batch size 20|batch size 30|batch size 40|Reference accuracy<br>(imagenet1k Top-5)|
|:----------------|------------:|------------:|------------:|------------:|---------------------------------------:|
|CaffeNet         |430          |496          |631          |716          |78.3%         |
|SqueezeNet       |608          |937          |1331         |1672         |78.8%         |
|NIN              |650          |902          |1062         |1222         |81.3%         |
|ResNet-18        |814          |1163         |1497         |1853         |88.7%         |
|ResNet-34        |1127         |1619         |2094         |2598         |91.0%         |
|Inception-BN     |1007         |1569         |2212         |2772         |90.8%         |
|ResNet-50        |1875         |3080         |4265         |5483         |92.6%         |
|Inception-v3     |2075         |3509         |4944         |6383         |93.3%         |
|VGG16            |1738         |2960         |4751         |5977         |89.8%         |
|ResNet-101       |2791         |4576         |6341         |8158         |93.3%         |
|VGG19            |1920         |3242         |5133         |6458         |89.8%         |
|ResNet-152       |3790         |6296         |8777         |11330        |93.1%         |
|ResNet-200       |2051         |2769         |3471         |4201         |unknown       |
|ResNeXt-50       |2248         |3863         |5468         |7089         |93.3%         |
|ResNeXt-101      |3350         |5749         |8126         |10539        |94.1%         |
|ResNeXt-101-64x4d|5140         |8679         |out of memory|out of memory|94.3%         |



## Available optimizers

- SGD
- DCASGD
- NAG
- SGLD
- Adam
- AdaGrad
- RMSProp
- AdaDelta
- Ftrl
- Adamax
- Nadam

To use these optimizers, specify the optimizer name in lowercase in `config.yml`.



## Utilities

### util/counter.sh

Count the number of files in each subdirectory.

```
$ util/counter.sh testdir
testdir contains 4 directories
Leopards    197
Motorbikes  198
airplanes   199
watch       200
```

### util/move_images.sh

Move the specified number of jpeg images from the target directory to the output directory while maintaining the directory structure.

```
$ util/move_images.sh 20 testdir newdir
processing Leopards
processing Motorbikes
processing airplanes
processing watch
$ util/counter.sh newdir
newdir contains 4 directories
Leopards    20
Motorbikes  20
airplanes   20
watch       20
```

### Prepare sample images for fine-tuning

Download [Caltech 101] dataset, and split part of it into the `example_images` directory.

```
$ util/caltech101_prepare.sh
```

- `example_images/train` is train set of 60 images for each classes
- `example_images/valid` is validation set of 20 images for each classes
- `example_imags/test` is test set of 20 images for each classes

```
$ util/counter.sh example_images/train
example_images/train contains 10 directories
Faces       60
Leopards    60
Motorbikes  60
airplanes   60
bonsai      60
car_side    60
chandelier  60
hawksbill   60
ketch       60
watch       60
```

With this data you can immediately try fine-tuning.

```
$ util/caltech101_prepare.sh
$ rm -rf images
$ mv exmaple_images images
$ docker-compose run finetuner
```



## Misc

### Use DenseNet

ImageNet pretrained DenseNet-169 model is introduced on [A MXNet implementation of DenseNet with BC structure].

You can use this model as below

#### Download parameter and symbol files

densenet-imagenet-169-0-0125.params
https://drive.google.com/open?id=0B_M7XF_l0CzXX3V3WXJoUnNKZFE

densenet-imagenet-169-0-symbol.json
https://raw.githubusercontent.com/bruinxiong/densenet.mxnet/master/densenet-imagenet-169-0-symbol.json

#### Rearrange downloaded files

Change the name of the downloaded files and store it as below

```
model/imagenet1k-densenet-169-0000.params
model/imagenet1k-densenet-169-symbol.json
```

#### Modify config

To use DenseNet-169 pretrained models, specify the `imagenet1k-densenet-169` in `config.yml`.




## Try image classification with jupyter notebook

```
$ docker-compose run --service-ports finetuner jupyter
```
*Please note that the `--service-port` option is required*

Replace the IP address of the displayed URL with the IP address of the host machine and access it from the browser.

Open the `classify_example/classify_example.ipynb` and try out the image classification sample using the VGG-16 pretrained model pretrained with ImageNet.



## config.yml options

### common settings

```
common:
  num_threads: 4
  # gpus: 0  # list of gpus to run, e.g. 0 or 0,2,5.
```

### data settings

train, validation and test RecordIO data generation settings.

mxnet-finetuner pack the image files into a recordIO file for increased performance.

```
data:
  quality: 95
  shuffle: 1
  center_crop: 0
  # use_japanese_label: 1
```

## finetune settings

```
finetune:
  models:  # specify models to use
    - imagenet1k-nin
    # - imagenet1k-inception-v3
    # - imagenet1k-vgg16
    # - imagenet1k-resnet-50
    # - imagenet11k-resnet-152
    # - imagenet1k-resnext-101
    # etc
  optimizers:  # specify optimizers to use
    - sgd
    # optimizers: sgd, dcasgd, nag, sgld, adam, adagrad, rmsprop, adadelta, ftrl, adamax, nadam
  num_epochs: 10  # max num of epochs
  # load_epoch: 0  # specify when using user fine-tuned model
  lr: 0.0001  # initial learning rate
  lr_factor: 0.1  # the ratio to reduce lr on each step
  lr_step_epochs: 10  # the epochs to reduce the lr, e.g. 30,60
  mom: 0.9  # momentum for sgd
  wd: 0.00001  # weight decay for sgd
  batch_size: 10  # the batch size
  disp_batches: 10  # show progress for every n batches
  # top_k: 0  # report the top-k accuracy. 0 means no report.
  # random_crop: 0  # if or not randomly crop the image
  # random_mirror: 0  # if or not randomly flip horizontally
  # max_random_h: 0  # max change of hue, whose range is [0, 180]
  # max_random_s: 0  # max change of saturation, whose range is [0, 255]
  # max_random_l: 0  # max change of intensity, whose range is [0, 255]
  # max_random_aspect_ratio: 0  # max change of aspect ratio, whose range is [0, 1]
  # max_random_rotate_angle: 0  # max angle to rotate, whose range is [0, 360]
  # max_random_shear_ratio: 0  # max ratio to shear, whose range is [0, 1]
  # max_random_scale: 1  # max ratio to scale
  # min_random_scale: 1  # min ratio to scale, should >= img_size/input_shape. otherwise use --pad-size
  # rgb_mean: '123.68,116.779,103.939'  # a tuple of size 3 for the mean rgb
  # monitor: 0  # log network parameters every N iters if larger than 0
  # pad_size: 0  # padding the input image
  auto_test: 1  # if or not test with validation data after fine-tuneing is completed
  train_accuracy_graph_output: 1
  # train_accuracy_graph_fontsize: 12
  # train_accuracy_graph_figsize: 8,6
  # train_accuracy_graph_slack_upload: 1
  # train_accuracy_graph_slack_channels:
  #   - general
```

### test settings

```
test:
  use_latest: 1  # Use last trained model. If set this option, model_prefix and model_epoch are ignored
  model_prefix: 201705292200-imagenet1k-nin-sgd
  model_epoch: 1
  # model_epoch_up_to: 10  # test from model_epoch to model_epoch_up_to respectively
  test_batch_size: 10
  # top_k: 10
  classification_report_output: 1
  # classification_report_digits: 3
  confusion_matrix_output: 1
  # confusion_matrix_fontsize: 12
  # confusion_matrix_figsize: 16,12
  # confusion_matrix_slack_upload: 1
  # confusion_matrix_slack_channels:
  #   - general
```


# Acknowledgement

- [MXNet]
- [Mo - Mustache Templates in Bash]
- [A MXNet implementation of DenseNet with BC structure]


# Licnese

[Apache-2.0] license.


[Apache-2.0]: https://github.com/dmlc/mxnet/blob/master/LICENSE
[MXNet]: https://github.com/apache/incubator-mxnet
[MXNet - Image Classification - Pre-trained Models]: https://github.com/apache/incubator-mxnet/tree/master/example/image-classification#pre-trained-models
[Mo - Mustache Templates in Bash]: https://github.com/tests-always-included/mo
[A MXNet implementation of DenseNet with BC structure]: https://github.com/bruinxiong/densenet.mxnet
[MXNet model gallery]: https://github.com/dmlc/mxnet-model-gallery
[Reproduce ResNet-v2 using MXNet]: https://github.com/tornadomeet/ResNet
[Caltech 101]: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
