# Available pretrained models

Classification accuracy of available pretrained models.
(Datasets are ImageNet1K, ImageNet11K and Place365 Challenge)

*Please note that the following results are calculated with different datasets.*
- ResNet accuracy from [Reproduce ResNet-v2 using MXNet]
- DenseNet-169 accuracy from [A MXNet implementation of DenseNet with BC structure]
- SE-ResNeXt-50 accuracy from [SENet.mxnet]
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
|ResNet-200                     |77.9%         |93.8%         |248MB        |240MB             |ImageNet1K |224x224    |
|Inception-v3                   |76.9%         |93.3%         |92MB         |84MB              |ImageNet1K |299x299    |
|ResNeXt-50                     |76.9%         |93.3%         |96MB         |89MB              |ImageNet1K |224x224    |
|ResNeXt-101                    |78.3%         |94.1%         |170MB        |162MB             |ImageNet1K |224x224    |
|ResNeXt-101-64x4d              |79.1%         |94.3%         |320MB        |312MB             |ImageNet1K |224x224    |
|ResNet-152 (imagenet11k)       |41.6%         |-             |311MB        |223MB             |ImageNet11K|224x224    |
|ResNet-50 (Place365 Challenge) |31.1%         |-             |181MB        |90MB              |Place365ch |224x224    |
|ResNet-152 (Place365 Challenge)|33.6%         |-             |313MB        |223MB             |Place365ch |224x224    |
|DenseNet-169                   |75.3%         |92.8%         |55MB         |48MB              |ImageNet1K |224x224    |
|SE-ResNeXt-50                  |76.7%         |93.4%         |103MB        |95MB              |ImageNet1K |224x224    |

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
|DenseNet-169                   |imagenet1k-densenet-169          |
|SE-ResNeXt-50                  |imagenet1k-se-resnext-50         |


[Reproduce ResNet-v2 using MXNet]: https://github.com/tornadomeet/ResNet
[A MXNet implementation of DenseNet with BC structure]: https://github.com/bruinxiong/densenet.mxnet
[SENet.mxnet]: https://github.com/bruinxiong/SENet.mxnet
[MXNet model gallery]: https://github.com/dmlc/mxnet-model-gallery
[MXNet - Image Classification - Pre-trained Models]: https://github.com/apache/incubator-mxnet/tree/master/example/image-classification#pre-trained-models
