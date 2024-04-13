import torch
import torchaudio
import torch.nn as nn
import unittest
from ChannelAttention import ChannelAttention

class GLFB(nn.Module):
    """
    Global Local Former Block implemented as described in 'A Mask Free Neural Network 
    for Monaural Speech Enhancement' Liu et, al.
    The paper does not fully describe all aspects of the implementation however, so 
    some assumptions were made and are identified in the comments below.
    """
    def __init__(self, in_channels, reduction_ratio, frame_size, n_frames):
        super(GLFB, self).__init__()
        self.glfb_layer_1 = nn.Sequential(
            # Layer Norm
            nn.LayerNorm((in_channels, frame_size, n_frames)),
            nn.Conv2d(  # Point Conv 1 & 3 double number of channels (https://arxiv.org/pdf/2306.04286.pdf Section 2.4)
                in_channels = in_channels,
                out_channels = in_channels * 2,
                kernel_size = 1
            ),
            # Depth Wise Seperable Convolution (https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/)
            nn.Conv2d( # From what I can tell no mention is made as to the kernel size or padding used for the depth wise seperable convolution
                in_channels = in_channels * 2,
                out_channels = in_channels * 2,
                kernel_size = 1,
                groups = in_channels * 2,
            ),
            # Gate (Not many details are given in the paper, so we'll assume a simple sigmoid gate for now) 
            nn.Conv2d(  # Gating is stated to halve the number of channels (https://arxiv.org/pdf/2306.04286.pdf Section 2.4)
                in_channels = in_channels * 2, 
                out_channels = in_channels, 
                kernel_size = 1
            ),
            nn.Sigmoid(),
            # Channel Attention
            ChannelAttention(in_channels = in_channels, reduction_ratio = reduction_ratio),
            nn.Conv2d(  # Point Conv 2 & 4 maintain number of channels (https://arxiv.org/pdf/2306.04286.pdf Section 2.4)
                in_channels = in_channels,
                out_channels = in_channels,
                kernel_size = 1
            )
        )
        self.glfb_layer_2 = nn.Sequential(
            # Layer Norm
            nn.LayerNorm((in_channels, frame_size, n_frames)),
            nn.Conv2d(  # Point Conv 1 & 3 double number of channels (https://arxiv.org/pdf/2306.04286.pdf Section 2.4)
                in_channels = in_channels,
                out_channels = in_channels * 2,
                kernel_size = 1
            ),
            # Gate (Not many details are given in the paper, so we'll assume a simple sigmoid gate for now) 
            nn.Conv2d(  # Gating is stated to halve the number of channels (https://arxiv.org/pdf/2306.04286.pdf Section 2.4)
                in_channels = in_channels * 2,
                out_channels = in_channels, 
                kernel_size = 1
            ),
            nn.Sigmoid(),
            nn.Conv2d(  # Point Conv 2 & 4 maintain number of channels (https://arxiv.org/pdf/2306.04286.pdf Section 2.4)
                in_channels = in_channels,
                out_channels = in_channels,
                kernel_size = 1
            )
        )
        
    def forward(self, x):
        l1 = self.glfb_layer_1(x)
        y = torch.add(l1, x)
        l2 = self.glfb_layer_2(y)
        out = torch.add(l2, y)
        return out
        
class TestGLFB(unittest.TestCase):
    def test_glfb(self):
        glfb = GLFB(in_channels = 1, reduction_ratio = 8)
        x = torch.randn(1, 320, 999)
        out = glfb(x)
        print(out.size())
        pass

if __name__ == "__main__":
    unittest.main()