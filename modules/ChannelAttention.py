import torch
import torch.nn as nn
import unittest

# referenced https://paperswithcode.com/method/channel-attention-module
class ChannelAttention(nn.Module):
    """
    Our implementation of Channel Attention as described in 'CBAM: Convolutional Block
    Attention Module' Woo et, al.
    """
    def __init__(self, in_channels, reduction_ratio):
        super(ChannelAttention, self).__init__()
        # Takes input from (N, C, H, W) and outputs (N, C, 1, 1) in both cases
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP part for the channel attention
        # I removed reduction ratio here since it was clipping the shapes
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features = in_channels,
                out_features = in_channels // reduction_ratio
            ),
            nn.ReLU(),
            nn.Linear(
                in_features = in_channels // reduction_ratio,
                out_features = in_channels
            ),
        )

        # Activation function
        self.activation = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.mlp(self.max_pool(x).squeeze(-1).squeeze(-1))
        cat_outs = torch.add(avg_out, max_out)
        out = self.activation(cat_outs).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return x * out
    
class TestChannelAttention(unittest.TestCase):
    def test_channel_attention(self):
        pass

if __name__ == '__main__':
    unittest.main()