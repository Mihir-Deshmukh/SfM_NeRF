import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class NeRFmodel(nn.Module):
    def __init__(self, embed_pos_L = 39):
        super(NeRFmodel, self).__init__()
        #############################
        # network initialization
        #############################
        self.layer1 = nn.Linear(embed_pos_L, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 4)

    # def position_encoding(self, x, L):
    #     #############################
    #     # Implement position encoding here
    #     #############################

    #     return y

    def forward(self, x):
        #############################
        # network structure
        #############################
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        output = self.layer3(x)     

        return x
