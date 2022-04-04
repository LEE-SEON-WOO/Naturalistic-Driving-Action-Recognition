# Naturalistic-Driving-Action-Recognition
Aicity 2022 Challenge Deep Learning Model Self Supervised(contrastive) learning

This repo covers an reference implementation for the following papers in PyTorch, using [AICITY2022 Challenge-Track3](https://www.aicitychallenge.org/2022-challenge-tracks/) as an illustrative example:
(1) Supervised Contrastive Learning. [Paper](https://arxiv.org/abs/2004.11362)  
(2) A Simple Framework for Contrastive Learning of Visual Representations. [Paper](https://arxiv.org/abs/2002.05709) 

## Dataset Preperation

The AICity2022 dataset can be downloaded from its [official website](https://www.aicitychallenge.org/)

## Experiment Comparsion
Results on AICITY2022:
|          |Arch | Setting | Loss | Accuracy(%) |
|----------|:----:|:---:|:---:|:---:|
|  CrossEntropy | ResNet50-C3D | Contrastive   | Cross Entropy |  00.0  |
|  NCELoss     | ResNet50 | Contrastive   | Contrastive   |  00.0  | 

## Running
You might use `config.py` to set proper setting, and/or switch to `datasets/test_dataset.py` and `datasets/train_dataset.py`.  
**Train**
```
python main.py --mode train
```
**Test**  
```
python main.py --mode test
```

Linear evaluation stage:
```
python main_classifier.py --ckpt /checkpoints/model.pth
```

### Pretrained weights
```bibtex
@inproceedings{hara3dcnns,
  author={Kensho Hara and Hirokatsu Kataoka and Yutaka Satoh},
  title={Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={6546--6555},
  year={2018},
}
```
Pre-trained models are available [here](https://drive.google.com/open?id=1xbYbZ7rpyjftI_KCk6YuL-XrfQDz7Yd4).
All models are trained on Kinetics-700 (_K_), Moments in Time (_M_), STAIR-Actions (_S_), or merged datasets of them (_KM_, _KS_, _MS_, _KMS_).  
```misc
r3d18_K_200ep.pth: --model resnet --model_depth 18 --n_pretrain_classes 700
r3d18_KM_200ep.pth: --model resnet --model_depth 18 --n_pretrain_classes 1039
r3d34_K_200ep.pth: --model resnet --model_depth 34 --n_pretrain_classes 700
r3d34_KM_200ep.pth: --model resnet --model_depth 34 --n_pretrain_classes 1039
r3d50_K_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 700
r3d50_KM_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 1039
r3d50_KMS_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 1139
r3d50_KS_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 800
r3d50_M_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 339
r3d50_MS_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 439
r3d50_S_200ep.pth: --model resnet --model_depth 50 --n_pretrain_classes 100
r3d101_K_200ep.pth: --model resnet --model_depth 101 --n_pretrain_classes 700
r3d101_KM_200ep.pth: --model resnet --model_depth 101 --n_pretrain_classes 1039
r3d152_K_200ep.pth: --model resnet --model_depth 152 --n_pretrain_classes 700
r3d152_KM_200ep.pth: --model resnet --model_depth 152 --n_pretrain_classes 1039
r3d200_K_200ep.pth: --model resnet --model_depth 200 --n_pretrain_classes 700
r3d200_KM_200ep.pth: --model resnet --model_depth 200 --n_pretrain_classes 1039
```

## Reference
Please cite the following article if you use this code or pre-trained models:

```
@misc{seonwoolee2022naturalisticdrivingationrecognition,
    title   = {TAL-CoLR:Temporal Action Localization using Contrastive Learning Repesentation},
    author  = {Seon-Woo Lee, Jeong-Gu Kim, Heo-June Kim, Mun-Hyung Lee, So-Hee Yong, Jang-Woo Kwon},
    booktitle = {IEEE Xplore Digital Library and CVF Open Access},
    howpublished = {\url{https://github.com/LEE-SEON-WOO/Naturalistic-Driving-Action-Recognition/}},
    year    = {2022},
    note = {commit xxxxxxx}
}

```


## Acknowledgement
We refer to these code.  [codebase 1](https://github.com/okankop/Driver-Anomaly-Detection)and [codebase 2](https://github.com/HobbitLong/CMC), which we build our work on top.


### Attributions/Thanks
- This project could not have happened without the advice given by our Contributors(HCI-LAB Member). 
  - This project also borrowed some code from the lists below.
    - [SupContrast](https://github.com/HobbitLong/SupContrast)
    - [kenshohara](https://github.com/kenshohara/3D-ResNets-PyTorch)
    - [DAVEISHAN](https://github.com/DAVEISHAN/TCLR)
    - [okankop](https://github.com/okankop/Driver-Anomaly-Detection)
    - [dddzg](https://github.com/dddzg/MoCo)
- Some readme/docs formatting was borrowed from Cheng-Bin Jin's [Cheng-Bin Jin Style](https://github.com/ChengBinJin/semantic-image-inpainting)

### Reference
- 

