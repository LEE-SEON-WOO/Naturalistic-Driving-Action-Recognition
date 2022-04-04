"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from prop_model import R3D_MLP
class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, feature_dim, model_depth, opt, num_classes=18):
        super(SupCEResNet, self).__init__()
        #NormalClass(1)+AnomalClass(17) = 18 Class
        self.encoder = R3D_MLP(feature_dim=feature_dim, 
                            model_depth=model_depth, 
                            opt=opt)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))

class LinearClassifier(nn.Module):
    """Just Linear classifier"""
    def __init__(self, feature_dim, model_depth, opt, num_classes=18):
        super(LinearClassifier, self).__init__()
        _, feat_dim = R3D_MLP(feature_dim=feature_dim, 
                            model_depth=model_depth, 
                            opt=opt)
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


class Fusion_R3D(nn.Module):
    def __init__(self, feature_dim, model_depth, opt, num_classes=18):
        super(LinearClassifier, self).__init__()
        pass
    
    def forward(self, x):
        pass
        