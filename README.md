# CapProNet_tf
# CapProNet with backbone of ResNet in TensorFlow

This is the offical implementation of paper [CapProNet: Deep Feature Learning via Orthogonal Projections onto Capsule Subspaces](https://arxiv.org/abs/1805.07621) with backbone of Resnet model for ImageNet database. The backbone network has been borrowd from Tensorflow offical implementation of resnet [here](https://github.com/tensorflow/models/tree/master/official/resnet). 
See the following papers for more detailes about the backbone architecture:

[1] [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

[2] [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

In code v1 refers to the resnet defined in [1], while v2 correspondingly refers to [2]. The principle difference between the two versions is that v1 applies batch normalization and activation after convolution, while v2 applies batch normalization, then activation, and finally convolution. A schematic comparison is presented in Figure 1 (left) of [2].

Please proceed according to which dataset you would like to train/evaluate on:



### Setup

We used Tensorflow version 1.8 and Python 2.7 for our experiments
First make sure you've [added the models folder to your Python path](/official/#running-the-models); otherwise you may encounter an error like `ImportError: No module named official.resnet`.

## ImageNet

### Setup
To begin, you will need to download the ImageNet dataset and convert it to TFRecord format. Follow along with the [Inception guide](https://github.com/tensorflow/models/tree/master/research/inception#getting-started) in order to prepare the dataset.

Once your dataset is ready, you can begin training the model as follows:

```
python caps_imagenet_main.py --data_dir=/path/to/imagenet
```

The model will begin training and will automatically evaluate itself on the validation data roughly once per epoch.

Note that there are a number of other options you can specify, including `--model_dir` to choose where to store the model and `--resnet_size` to choose the model size (options include ResNet-18 through ResNet-200). See [`resnet.py`](resnet.py) for the full list of options.


## Compute Devices
Training is accomplished using the DistributionStrategies API. (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md)

The appropriate distribution strategy is chosen based on the `--num_gpus` flag. By default this flag is one if TensorFlow is compiled with CUDA, and zero otherwise.

num_gpus:
+ 0:  Use OneDeviceStrategy and train on CPU.
+ 1:  Use OneDeviceStrategy and train on GPU.
+ 2+: Use MirroredStrategy (data parallelism) to distribute a batch between devices.
