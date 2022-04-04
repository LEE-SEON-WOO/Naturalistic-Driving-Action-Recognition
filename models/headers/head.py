import torch.nn as nn
import torch.nn.functional as F

"""
This code Header Code

Projection

"""
class ProjectionHead(nn.Module):
    def __init__(self, output_dim, model_depth):
        super(ProjectionHead, self).__init__()
        if model_depth == 18:
            self.hidden = nn.Linear(512, 256)
        elif model_depth == 50:
            self.hidden = nn.Linear(2048, 256)
        elif model_depth == 101:
            self.hidden = nn.Linear(2048, 256)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Linear(256, output_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.out(x)
        x = F.normalize(x, p=2, dim=1)

        return x


