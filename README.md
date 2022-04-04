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
 
R3D-18 with UCF101 pretraining: [Google Drive](https://drive.google.com/file/d/1Y-YmohPPeZKmd8MO_KVYKDNoIbzpjQWV/view?usp=sharing)<br/>R3D-18 with Kinetics400 pretraining: [Google Drive](https://drive.google.com/file/d/1m-u8N18dYFqP9B2JF3dEYOowKg3xDrds/view?usp=sharing)<br/>R2+1D-18 with Kinetics400 pretraining: [Google Drive](https://drive.google.com/file/d/1cuM4vFJA8wDDYmkQeAhwBUDQD0aDGmqD/view?usp=sharing)

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

