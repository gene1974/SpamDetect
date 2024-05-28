import torch
import torch.nn as nn

class BatchNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        '''
        x: (batch_size, max_sen_len, dim)
        '''
        mean = torch.mean(x, dim = 0, keepdim = True) # (max_sen_len, dim)
        std = torch.std(x, dim = 0, keepdim = True)
        x = (x - mean) / (std + 1e-7)
        x = self.gamma * x + self.beta
        return x

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        '''
        input:
            x: (batch_size, max_sen_len, dim)
        '''
        mean = torch.mean(x, dim = -1, keepdim = True) # (batch_size, sen_len, 1)
        std = torch.std(x, dim = -1, keepdim = True)
        x = (x - mean) / (std + 1e-7)
        x = self.g * x + self.b
        return x


