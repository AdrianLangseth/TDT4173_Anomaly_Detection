import torch
import torch.nn.functional as F
from torch.nn import Linear

class NN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.out = Linear(hidden_size, output_size)
        
    def forward(self, x):
        hidden = F.relu(self.fc1(x))
        output = self.out(hidden)
        return output