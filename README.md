# mxnet-finetuner

An all-in-one Deep Learning toolkit for image classification to fine-tuning pretrained models using MXNet.


## Prerequisites

- docker
- docker-compose
- jq
- wget or curl

When using NVIDIA GPUs

- nvidia-docker (Both version 1.0 and 2.0 are acceptable)

If you are using nvidia-docker version 1.0 and have never been running the `nvidia-docker` command after installing it, run the following command at least once to create the volume for GPU container.

```
$ nvidia-docker run --rm nvidia/cuda nvidia-smi
```

## Setup

```
$ git clone https://github.com/knjcode/mxnet-finetuner
$ cd mxnet-finetuner
$ bash setup.sh
```

`setup.sh` will automatically generate` docker-compose.yml` and `config.yml` which are necessary for executing this tool according to your environment such as existence of the GPU. Please see [common settings](#common-settings) on how to run with GPU image on NVIDIA GPUs.

When updating the GPU driver of the host machine, re-running `setup.sh`. Please see [After updating the GPU driver of the host machine](docs/setup.md#after-updating-the-gpu-driver-of-the-host-machine) for details.

## Example usage


### 1. Arrange images into their respective directories

A training data directory (`images/train`), validation data directory (`images/valid`), and test data directory (`images/test`) should containing one subdirectory per image class.

For example, arrange training, validation, and test data as follows.

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
    test/
        airplanes/
            airplane201.jpg
            airplane202.jpg
            ...
        watch/
            watch201.jpg
            watch202.jpg
            ...
```


### 2. Edit config.yml

Edit `config.yml` as you like.

For example
```
common:
  num_threads: 4
  gpus: 0

data:
  quality: 100
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
  lr_step_epochs: 10,20
  mom: 0.9
  wd: 0.00001
  batch_size: 10
```

Please see [common settings](#common-settings) on how to run with GPU image on NVIDIA GPUs.

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
- Make training accuracy/loss graph
- Make confusion matrix
- Upload training accuracy/loss graph and confusion matrix to Slack

Training accuracy/loss graph and/or confusion matrix are save at `logs/` directory.  
Trained models are save at `model/` directory.

Trained models are saved with the following file name for each epoch.
```
model/201705292200-imagenet1k-nin-sgd-0000.params
```

If you want to upload results to Slack, set `SLACK_API_TOKEN` environment variable and edit `config.yml` as below.
```
finetune:
  train_accuracy_graph_slack_upload: 1
  train_loss_graph_slack_upload: 1
test:
  confusion_matrix_slack_upload: 1
```


### 4. Predict with trained models

Select the trained model and epoch you want to use for testing and edit `config.yml`

If you want to use `model/201705292200-imagenet1k-nin-sgd-0001.params`, edit `config.yml` as blow.

```
test:
  model: 201705292200-imagenet1k-nin-sgd-0001
```

When you want to use the latest highest validation accuracy trained model, edit `config.yml` as below.

```
test:
  use_latest: 1
```

If set this option, `model` is ignored.

When you are done, you can predict with the following command

```
$ docker-compose run finetuner test
```

Predict result and classification report and/or confusion matrix are save at `logs/` directory.


## Available pretrained models

|model                          |pretrained model name            |
|:------------------------------|:--------------------------------|
|CaffeNet                       |imagenet1k-caffenet              |
|SqueezeNet                     |imagenet1k-squeezenet            |
|NIN                            |imagenet1k-nin                   |
|VGG16                          |imagenet1k-vgg16                 |
|Inception-BN                   |imagenet1k-inception-bn          |
|ResNet-50                      |imagenet1k-resnet-50             |
|ResNet-152                     |imagenet1k-resnet-152            |
|Inception-v3                   |imagenet1k-inception-v3          |
|DenseNet-169                   |imagenet1k-densenet-169          |
|SE-ResNeXt-50                  |imagenet1k-se-resnext-50         |

To use these pretrained models, specify the following pretrained model name in `config.yml`.

For details, please check [Available pretrained models](docs/pretrained_models.md)


## Available optimizers

- SGD
- NAG
- RMSProp
- Adam
- AdaGrad
- AdaDelta
- Adamax
- Nadam
- DCASGD
- SGLD
- Signum
- FTML
- Ftrl

To use these optimizers, specify the optimizer name in lowercase in `config.yml`.


## Benchmark (Speed and Memory Footprint)

Single TITAN X (Maxwell) with batch size 40

|Model       |speed (images/sec)|memory (MiB)|
|:-----------|:-----------------|:-----------|
|CaffeNet    |1077.63           |716         |
|ResNet-50   |111.04            |5483        |
|Inception-V3|82.34             |6383        |
|ResNet-152  |48.28             |11330       |

For details, please check [Benchmark](docs/benchmark.md)


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

### How to freeze layers during fine-tuning

If you set the number of target layer to `finetune.num_active_layers` in `config.yml` as below, only layers whose number is not greater than the number of the specified layer will be train.

```
finetune:
  models:
    - imagenet1k-nin
  optimizers:
    - sgd
  num_active_layers: 6
```

The default for `finetune.num_active_layers` is `0`, in which case all layers are trained.

If you set `1` to `finetune.num_active_layers`, only the last fully-connected layers are trained.

You can check the layer numbers of various pretrained models with `num_layers` command.

```
$ docker-compose run finetuner num_layers <pretrained model name>
```

For details, please check [How to freeze layers during fine-tuning](docs/freeze_layers.md)


### Training from scratch

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

For details, please check [Available models training from scratch](docs/train_from_scratch.md)


## Averaging ensemble test with trained models

You can do averaging ensemble test using multiple trained models.

If you want to use the following the three trained models,

- `model/20180130074818-imagenet1k-nin-nadam-0003.params`
- `model/20180130075252-imagenet1k-squeezenet-nadam-0003.params`
- `model/20180131105109-imagenet1k-caffenet-nadam-0003.params`

edit `config.yml` as blow.

```
ensemble:
  models:
    - 20180130074818-imagenet1k-nin-nadam-0003
    - 20180130075252-imagenet1k-squeezenet-nadam-0003
    - 20180131105109-imagenet1k-caffenet-nadam-0003
```


When you are done, you can do averaging ensemble test with the following command.

```
$ docker-compose run finetuner ensemble test
```

If you want to use validation dataset, do as follows.

```
$ docker-compose run finetuner ensemble valid
```

Averaging ensemble test result and classification report and/or confusion matrix are save at `logs/` directory.


## Export your trained model

You can export your trained model in a format that can be used with [Model Server for Apache MXNet] as follows.

```
$ docker-compose run finetuner export
```

The exported file (extension is .model) is saved at `model/` directory.

Please check [export settings](#export-settings) for export settings.


## Serve your exported model

You can serve your exported model as API server.

With the following command, launch the API server with the last exported model using pre-configured Docker image of [Model Server for Apache MXNet].

```
$ docker-compose up -d mms
```

The API server is started at port 8080 of your local host.

Then you will `curl` a `POST` to the MMS predict endpoint with the test image. (For exmple, use `airlane.jpg`).

```
$ curl -X POST http://127.0.0.1:8080/model/predict -F "data=@airplane.jpg"
```

The predict endpoint will return a prediction response in JSON. It will look something like the following result:

```
{
  "prediction": [
    [
      {
        "class": "airplane",
        "probability": 0.9950716495513916
      },
      {
        "class": "watch",
        "probability": 0.004928381647914648
      }
    ]
  ]
}
```


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
  gpus: 0  # list of gpus to run, e.g. 0 or 0,2,5.
```

If a machine has one or more GPU cards installed, then each card is labeled by a number starting from 0.
To use GPU for training or inference, specify GPU number in common.gpus.

If you do not use the GPU or you can not use it, please comment out common.gpus.

In the environment where GPU can not be used, `common.gpus` in `config.yml` generated by `setup.sh` is automatically commented out.

### data settings

train, validation and test RecordIO data generation settings.

mxnet-finetuner resize and pack the image files into a recordIO file for increased performance.

By setting the `resize_short`, you can resize shorter edge of images to that size.

If `resize_short` is not specified, it is automatically determined according to the model you are using.

```
data:
  quality: 100
  shuffle: 1
  center_crop: 0
  # test_center_crop: 1
  # resize_short: 256
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
    # - imagenet1k-se-resnext-50
    # etc
  optimizers:  # specify optimizers to use
    - sgd
    # optimizers: sgd, nag, rmsprop, adam, adagrad, adadelta, adamax, nadam, dcasgd, signum, etc.
  # num_active_layers: 1  # train last n-layers without last fully-connected layer
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
  # data_aug_level: 3  # preset data augumentation level
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
  train_loss_graph_output: 1
  # train_loss_graph_fontsize: 12
  # train_loss_graph_figsize: 8,6
  # train_loss_graph_slack_upload: 1
  # train_loss_graph_slack_channels:
  #   - general
```

#### data_aug_level

By setting the `data_aug_level` parameter, you can set the data augumentation settings collectively.

|Level  |settings                          |
|:------|:---------------------------------|
|Level 1|random_crop: 1<br>random_mirror: 1|
|Level 2|max_random_h: 36<br>max_random_s: 50<br>max_random_l: 50<br>+ Level 1|
|Level 3|max_random_aspect_ratio: 0.25<br>max_random_rotate_angle: 10<br>max_random_shear_ratio: 0.1<br>+ Level 2|

If `data_aug_level` is set, parameters related to data augumentation will be overwritten.


### test settings

```
test:
  use_latest: 1  # Use last trained model. If set this option, model is ignored
  model: 201705292200-imagenet1k-nin-sgd-0001
  # model_epoch_up_to: 10  # test from epoch of model to model_epoch_up_to respectively
  test_batch_size: 10
  # top_k: 10
  # rgb_mean: '123.68,116.779,103.939'  # a tuple of size 3 for the mean rgb
  classification_report_output: 1
  # classification_report_digits: 3
  confusion_matrix_output: 1
  # confusion_matrix_fontsize: 12
  # confusion_matrix_figsize: 16,12
  # confusion_matrix_slack_upload: 1
  # confusion_matrix_slack_channels:
  #   - general
```


### ensemble settings

```
# ensemble settings
ensemble:
  models:
    - 20180130074818-imagenet1k-nin-nadam-0003
    - 20180130075252-imagenet1k-squeezenet-nadam-0003
    - 20180131105109-imagenet1k-caffenet-nadam-0003
  # weights: 1,1,1
  ensemble_batch_size: 10
  # top_k: 10
  # rgb_mean: '123.68,116.779,103.939'  # a tuple of size 3 for the mean rgb
  classification_report_output: 1
  # classification_report_digits: 3
  confusion_matrix_output: 1
  # confusion_matrix_fontsize: 12
  # confusion_matrix_figsize: 16,12
  # confusion_matrix_slack_upload: 1
  # confusion_matrix_slack_channels:
  #   - general
```


### export settings

```
# export settings
export:
  use_latest: 1 # Use last trained model. If set this option, model is ignored
  model: 201705292200-imagenet1k-nin-sgd-0001
  # top_k: 10  # report the top-k accuracy
  # rgb_mean: '123.68,116.779,103.939'  # a tuple of size 3 for the mean rgb
  # center_crop: 1  # if or not center crop at image preprocessing
  # model_name: model
```


# Acknowledgement

- [MXNet]
- [Mo - Mustache Templates in Bash]
- [A MXNet implementation of DenseNet with BC structure]
- [SENet.mxnet]
- [Model Server for Apache MXNet]

# Licnese

[Apache-2.0] license.


[Apache-2.0]: https://github.com/dmlc/mxnet/blob/master/LICENSE
[MXNet]: https://github.com/apache/incubator-mxnet
[Mo - Mustache Templates in Bash]: https://github.com/tests-always-included/mo
[A MXNet implementation of DenseNet with BC structure]: https://github.com/bruinxiong/densenet.mxnet
[Caltech 101]: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
[SENet.mxnet]: https://github.com/bruinxiong/SENet.mxnet
[Model Server for Apache MXNet]: https://github.com/awslabs/mxnet-model-server
