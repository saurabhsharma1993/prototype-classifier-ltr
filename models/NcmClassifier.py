import copy
import torch.nn as nn
from utils import *
from os import path
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class NcmClassifier(nn.Module):    
    def __init__(self, num_classes = 100, feat_dim = 64):
        super(NcmClassifier, self).__init__()       
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.register_parameter(name='T', param=torch.nn.Parameter(torch.ones(self.feat_dim).float()))                          # channel wise scaling
    def forward(self, x, centroids, stddevs, phase = 'train'):
        dists = torch.sqrt(torch.sum(torch.square(torch.div(x[:,None,:] - centroids[None,:],self.T[None,None])), dim=-1))       # channel wise scaling
        # scores are negative of the distances themselves.
        return -dists/2, None                         