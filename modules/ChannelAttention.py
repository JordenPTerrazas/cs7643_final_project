import torch
import torch.nn as nn
import unittest

# referenced https://paperswithcode.com/method/channel-attention-module
class ChannelAttention(nn.Module):
    """
    Our implementation of Simple Channel Attention, described by Liu. 
    """
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        # Takes input from (N, C, H, W) and outputs (N, C, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  

        # Takes avg poooled input
        self.ca = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = 1,
            padding = 0,
            stride = 1,
            groups = 1,
            bias = True
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.ca(x)
        return x
    
class TestChannelAttention(unittest.TestCase):
    def test_channel_attention(self):
        pass

if __name__ == '__main__':
    unittest.main()