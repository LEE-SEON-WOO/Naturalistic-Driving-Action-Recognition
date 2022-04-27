from matplotlib.colors import Normalize
import torch.nn as nn
import torch.nn.functional as F

"""
This code Header Code

Projection

"""
class ProjectionHead(nn.Module):
    def __init__(self, 
                output_dim, 
                model_depth,
                head='mlp'):
        super(ProjectionHead, self).__init__()
        self.head = head
        depth = {18:(512, 256), 50:(2048, 256), 101:(2048, 256)}
        if head=='linear':
            dim_in = depth[model_depth][0]
            self.hidden = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, output_dim),
                nn.Sigmoid()
            )
        elif head=='mlp':
            dim_in = depth[model_depth][0]
            self.hidden = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, output_dim)
                
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.hidden(x)
        if self.head == 'mlp':
            x = F.normalize(x, p=2, dim=1)
        return x
