import torch
import torch.nn as nn
import torch.nn.functional as F

class MapperModel(nn.Module):
    def __init__(self):
        """
        Basic Neural Network for starters
        """
        super(MapperModel, self).__init__()
        self.model = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 4), stride=(2, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(3, 4), stride=(2, 1)),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=191, out_features=128),
        )
    
    def forward(self, x):
        return self.model(x)