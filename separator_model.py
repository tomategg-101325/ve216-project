import torch
import torch.nn as nn
import numpy as np

import global_params as params
import global_functions as funcs

class SignalSeparator(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)
            # nn.Linear(input_size, 4)
        )
    
    def forward(self, x):
        return self.net(x)

