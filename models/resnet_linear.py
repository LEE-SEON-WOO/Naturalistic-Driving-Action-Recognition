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

from .prop_model import R3D_MLP
class Fusion_R3D(nn.Module):
    def __init__(self, 
                dash:R3D_MLP,
                rear:R3D_MLP,
                right:R3D_MLP, 
                feature=2048,
                n_classes=18,
                with_classifier:bool=False):
        super(Fusion_R3D, self).__init__()
        self.dash_model=dash
        self.rr_model=rear
        self.rt_model=right
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 2048))
        if with_classifier:
            self.classifier = nn.Linear(feature, n_classes)
            
    def forward(self, x):
        #nm = normal, an = anormal
        _, d_an = self.dash_model(x)
        _, rr_an = self.rr_model(x)
        #Projection(vector) positive, anomaly=negative
        _, rt_an = self.rt_model(x)
        
        x = torch.cat([d_an, rr_an, rt_an], axis=1)
        
        x = x.unsqueeze(dim=1)
        x = self.avgpool(x)
        x = x.squeeze(dim=1)
        
        if hasattr(self, 'classifier'):
            
            x = self.classifier(x)
        
        return x

if __name__ == '__main__':
    
    # print(model)
    
    
    from opts import parse_args
    # opt = parse_args()
    input = torch.rand(5, 3, 16, 112, 112).cuda()
    #r2p1d50_K_200ep.pth --model resnet --model_depth 50 --n_pretrain_classes 700 -> 93.4 (1st)

    
    inp = torch.rand(8, 3, 16, 112, 112).cuda()
    model = Fusion_R3D(dash=R3D_MLP(128, 50, opt=parse_args(pretrain_path='../checkpoints/best_model_resnet_Dashboard.pth')),
               rear=R3D_MLP(128, 50, opt=parse_args(pretrain_path='../checkpoints/best_model_resnet_Rear.pth')),
               right=R3D_MLP(128, 50, opt=parse_args(pretrain_path='../checkpoints/best_model_resnet_Right.pth')),
               with_classifier=True)
    print(model)
    #model = generate_model(opt, removed_classifier=True)
