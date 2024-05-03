import torch
from torch import nn
torch.manual_seed(42)
class ClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features= 10)
        self.layer_2 = nn.Linear(in_features=10, out_features= 10)
        self.layer_3 = nn.Linear(in_features=10, out_features= 1)
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self,x):
        return self.layer_3(self.relu(self.layer_2(self.tanh(self.layer_1(x)))))