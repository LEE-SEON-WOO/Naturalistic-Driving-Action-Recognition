import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

class MLP(nn.Module):
    def __init__(self, final_embedding_size = 128, use_normalization = True):
        super(MLP, self).__init__()

        self.final_embedding_size = final_embedding_size
        self.use_normalization = use_normalization
        self.fc1 = nn.Linear(512,512, bias = True)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, self.final_embedding_size, bias = False)
        self.temp_avg = nn.AdaptiveAvgPool3d((1,None,None))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                m.bias.data.fill_(0.01)

    def forward(self, x):
        with autocast():
            x = self.temp_avg(x)
            x = x.flatten(1)
            x = self.relu(self.bn1(self.fc1(x)))
            x = F.normalize(self.bn2(self.fc2(x)), p=2, dim=1)
            return x