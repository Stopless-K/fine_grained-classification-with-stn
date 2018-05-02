# ResNeXt & ResNet Pytorch Implementation
- ResNet (Deep Residual Learning for Image Recognition)
- Pre-act ResNet (Identity mappings in deep residual networks)
- ResNeXt (Aggregated Residual Transformations for Deep Neural Networks)
- DenseNet (Densely Connected Convolutional Networks)

- [x] Train on CIFAR-10 and CIFAR-100 with ResNeXt29-8-64d and ResNeXt29-16-64d
- [x] Train on CIFAR-10 and CIFAR-100 with ResNet20,32,44,56,110
- [x] Train on CIFAR-10 and CIFAR-100 with Pre-Activation ResNet20,32,44,56,110
- [x] Train on CIFAR-10 and CIFAR-100 with DenseNet
- [x] Train ImageNet

## Usage
To train on CIFAR-10 using 4 gpu:

```bash
python main.py ./data/cifar.python --dataset cifar10 --arch resnext29_8_64 --save_path ./snapshots/cifar10_resnext29_8_64_300 --epochs 300 --learning_rate 0.05 --schedule 150 225 --gammas 0.1 0.1 --batch_size 128 --workers 4 --ngpu 4
```

Or there are some off-the-shelf scripts can dirrectly be used for training.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 sh ./scripts/train_model.sh resnet20 cifar10
```

Train the ResNet-18 on ImageNet with 8 GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh ./scripts/train_imagenet.sh resnet18
```

A simplified CaffeNet-like model for CIFAR-10, which obtains the top1 accuracy of 89.5.

```
sh ./scripts/cifar10_caffe.sh
```

## Configurations
From the original [ResNeXt](https://arxiv.org/pdf/1611.05431.pdf) and [ResNet](https://arxiv.org/abs/1512.03385) papers:

| depth | cardinality | base width | parameters |  error   cifar10 |   error  cifar100 | architecture |
|:-----:|:-----------:|:----------:|:----------:|:----------------:|:-----------------:|:------------:|
|  29   |      8      |     64     |    34.4M   |       3.65       |       17.77       |   ResNeXt    |
|  29   |      16     |     64     |    68.1M   |       3.58       |       17.31       |   ResNeXt    |
|  20   |      *      |     *      |    0.27M   |       8.75       |         -         |   ResNet     |
|  32   |      *      |     *      |    0.46M   |       7.51       |         -         |   ResNet     |
|  44   |      *      |     *      |    0.66M   |       7.17       |         -         |   ResNet     |
|  56   |      *      |     *      |    0.85M   |       6.97       |         -         |   ResNet     |
| 110   |      *      |     *      |    1.7M    |       6.61       |         -         |   ResNet     |
| 1202  |      *      |     *      |   19.4M    |       7.93       |         -         |   ResNet     |

## My Results {Last Epoch Error (Best Error)}
| depth | cardinality | base width | parameters |  error   cifar10 |   error  cifar100 | architecture |
|:-----:|:-----------:|:----------:|:----------:|:----------------:|:-----------------:|:------------:|
|  29   |      8      |     64     |    34.4M   |       3.67       |    17.66(17.47)   |   ResNeXt    |
|  29   |      16     |     64     |    68.1M   |    3.59(3.39)    |    17.31(17.06)   |   ResNeXt    |
|  20   |      *      |     *      |    0.27M   |       8.47       |       32.99       |   ResNet     |
|  32   |      *      |     *      |    0.46M   |       7.67       |       30.80       |   ResNet     |
|  44   |      *      |     *      |    0.66M   |       7.23       |       29.45       |   ResNet     |
|  56   |      *      |     *      |    0.85M   |       6.86       |       28.89       |   ResNet     |
| 110   |      *      |     *      |    1.7M    |       6.62       |       27.62       |   ResNet     |
|  20   |      *      |     *      |    0.27M   |       8.35       |       31.79       |   Pre-Act    |
|  32   |      *      |     *      |    0.46M   |       7.57       |       30.02       |   Pre-Act    |
|  44   |      *      |     *      |    0.66M   |                  |       29.43       |   Pre-Act    |

#### ImageNet-1k (CenterCrop)
|    arch    | Top-1 Accuracy | Top-5 Accuracy |  Loss  |
|:----------:|:--------------:|:--------------:|:------:|
| ResNet-18  |      70.17     |     89.48      | 1.3097 |
| ResNet-18  |      70.22     |     89.43      | 1.5979 |
| ResNet-18  |      70.28     |     89.63      | 1.3023 |
| ResNet-34  |      73.92     |     91.62      | 1.0315 |
| ResNet-50  |      76.19     |     93.10      | 0.8172 |


## Other Projects
* [Torch (@facebookresearch)](https://github.com/facebookresearch/ResNeXt). (Original) CIFAR and ImageNet
* [MXNet (@dmlc)](https://github.com/dmlc/mxnet/tree/master/example/image-classification#imagenet-1k). ImageNet
* [PyTorch (@prlz77)](https://github.com/prlz77/ResNeXt.pytorch). CIFAR
* [EraseReLU](https://github.com/D-X-Y/EraseReLU). (will be public soon)

## Cite
```
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Computer Vision and Pattern Recognition},
  year={2016}
}
@inproceedings{he2016identity,
  title={Identity mappings in deep residual networks},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={European Conference on Computer Vision},
  year={2016}
}
@inproceedings{xie2017aggregated,
  title={Aggregated residual transformations for deep neural networks},
  author={Xie, Saining and Girshick, Ross and Doll{\'a}r, Piotr and Tu, Zhuowen and He, Kaiming},
  booktitle={Computer Vision and Pattern Recognition},
  year={2017}
}
@inproceedings{huang2017densely,
  title={Densely connected convolutional networks},
  author={Huang, Gao and Liu, Zhuang and Weinberger, Kilian Q and van der Maaten, Laurens},
  booktitle={Computer Vision and Pattern Recognition},
  year={2017}
}
@article{dong2017eraserelu,
  title={EraseReLU: A Simple Way to Ease the Training of Deep Convolution Neural Networks},
  author={Dong, Xuanyi and Kang, Guoliang and Zhan, Kun and Yang, Yi},
  journal={arXiv preprint arXiv:1709.07634},
  year={2017}
}
```

## Download the ImageNet dataset
The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) dataset has 1000 categories and 1.2 million images. The images do not need to be preprocessed or packaged in any database, but the validation images need to be moved into appropriate subfolders.

1. Download the images from http://image-net.org/download-images

2. Extract the training data:
  ```bash
  mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
  find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  cd ..
  ```

3. Extract the validation data and move images to subfolders:
  ```bash
  mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  ```
