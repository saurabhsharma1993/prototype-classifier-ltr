import torch.nn as nn
from utils import *
from os import path
import torch.nn.init as init

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class DotProduct_Classifier(nn.Module):
    
    def __init__(self, num_classes=1000, feat_dim=2048, *args):
        super(DotProduct_Classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
        self.apply(_weights_init)
        
    def forward(self, x, *args):
        x = self.fc(x)
        return x, None