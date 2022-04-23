
    

from models.resnet_linear import Fusion_R3D
from models.resnet_linear import Fusion_R3D
from models.prop_model import R3D_MLP
from data.cls_dataset import CLS
from opts import parse_args
import torch
if __name__ == '__main__':
    opt = parse_args()
    
    model = Fusion_R3D(dash=R3D_MLP(opt.feature_dim, opt.model_depth, 
                        opt=parse_args(pretrain_path='./checkpoints/best_model_resnet_Dashboard.pth')),
                        rear=R3D_MLP(opt.feature_dim, opt.model_depth, 
                        opt=parse_args(pretrain_path='./checkpoints/best_model_resnet_Rear.pth')),
                        right=R3D_MLP(opt.feature_dim, opt.model_depth, 
                        opt=parse_args(pretrain_path='./checkpoints/best_model_resnet_Right.pth')),
                        with_classifier=True)
    model = torch.nn.DataParallel(model)
    
    
    inp = torch.rand(64, 1, 16, 112, 112).cuda()
    
    
    output = model(inp, inp, inp)
    print(output.shape)