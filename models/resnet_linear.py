"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .prop_model import R3D_MLP


class LinearClassifier(nn.Module):
    """Just Linear classifier"""
    def __init__(self, model:R3D_MLP, feature_dim, num_classes=18):
        super(LinearClassifier, self).__init__()
        
        self.model = model
        self.fc = nn.Linear(feature_dim, num_classes)

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
        # self.avgpool = nn.AdaptiveAvgPool3d(None, 1, 1)
        if with_classifier:
            self.classifier = nn.Linear(feature, n_classes)
            
    def forward(self, x_dash, x_rear, x_right):
        #nm = normal, an = anormal
        bs = x_dash.shape[0]
        d_nm, d_an = self.dash_model(x_dash)
        d_nm = d_nm.unsqueeze(dim=1)
        
        d_an = d_an.unsqueeze(dim=1)
        d_an = d_an.permute(0, 2, 1)
        d_mat = torch.bmm(d_an, d_nm)
        
        #Projection(vector) positive, anomaly=negative
        r_nm, rr_an = self.rr_model(x_rear)
        r_nm = r_nm.unsqueeze(dim=1)
        
        rr_an = rr_an.unsqueeze(dim=1)
        rr_an = rr_an.permute(0, 2, 1)
        r_mat = torch.bmm(rr_an, r_nm)
        
        rt_nm, rt_an = self.rt_model(x_right)
        rt_an = rt_an.unsqueeze(dim=1)
        rt_an = rt_an.permute(0, 2, 1)
        
        rt_nm = rt_nm.unsqueeze(dim=1)
        
        rt_mat = torch.bmm(rt_an, rt_nm)
        x = torch.cat([d_mat, r_mat, rt_mat], axis=2)
        
        x = x.unsqueeze(dim=1) #
        # print(x.shape) #[2048, 1, 384]
        x = self.avgpool(x)
        # print(x.shape) #([2048, 1, 2048])
        x = x.reshape(bs, -1)
        
        # print(x.shape) #([2048, 2048])
        
        if hasattr(self, 'classifier'):
            
            x = self.classifier(x)
        
        x = F.softmax(x, dim=1)
        
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
    # model = generate_model(opt, removed_classifier=True)
