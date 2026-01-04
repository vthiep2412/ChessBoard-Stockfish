import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config

class ResBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ChessResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_residual_blocks = Config.NUM_RESIDUAL_BLOCKS
        self.num_filters = Config.NUM_FILTERS
        
        # Input: 18 x 8 x 8
        self.conv_input = nn.Conv2d(Config.INPUT_SHAPE[0], self.num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(self.num_filters)
        
        # Residual Tower
        self.res_blocks = nn.ModuleList([
            ResBlock(self.num_filters) for _ in range(self.num_residual_blocks)
        ])
        
        # Policy Head
        # Output: 4096 (64x64 simplified move space)
        self.conv_policy = nn.Conv2d(self.num_filters, 2, kernel_size=1, bias=False)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * 8 * 8, 4096)
        
        # Value Head
        # Output: 1 (Scalar evaluation: -1 to 1)
        self.conv_value = nn.Conv2d(self.num_filters, 1, kernel_size=1, bias=False)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(1 * 8 * 8, 256)
        self.fc_value2 = nn.Linear(256, 1)

    def forward(self, x):
        # Initial Conv
        out = F.relu(self.bn_input(self.conv_input(x)))
        
        # Res Blocks
        for res_block in self.res_blocks:
            out = res_block(out)
        
        # Policy Head
        p = F.relu(self.bn_policy(self.conv_policy(out)))
        p = p.view(-1, 2 * 8 * 8)
        p = self.fc_policy(p) # Logits
        # Note: We return logits here. Softmax applies in loss function or MCTS.
        
        # Value Head
        v = F.relu(self.bn_value(self.conv_value(out)))
        v = v.view(-1, 1 * 8 * 8)
        v = F.relu(self.fc_value1(v))
        v = torch.tanh(self.fc_value2(v))
        
        return p, v
