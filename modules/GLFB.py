import torch
import torchaudio
import torch.nn as nn
import unittest
from .ChannelAttention import ChannelAttention


class GLFB(nn.Module):
    """
    Global Local Former Block implemented as described in 'A Mask Free Neural Network 
    for Monaural Speech Enhancement' Liu et, al.
    The paper does not fully describe all aspects of the implementation however, so 
    some assumptions were made and are identified in the comments below.
    """
    def __init__(self, in_channels, features, signals):
        super(GLFB, self).__init__()
        
        # Different LayerNorms so that different parameters can be learned
        self.norm_1 = nn.LayerNorm([in_channels, features, signals])
        self.norm_2 = nn.LayerNorm([in_channels, features, signals])
        
        # Point Conv 1 & 3 double number of channels (https://arxiv.org/pdf/2306.04286.pdf Section 2.4)
        # Point Conv 2 & 4 maintain number of channels (https://arxiv.org/pdf/2306.04286.pdf Section 2.4)
        self.point_conv_1 = nn.Conv2d(  
            in_channels = in_channels,
            out_channels = in_channels * 2,
            kernel_size = 1
        ) 
        self.point_conv_2 = nn.Conv2d(  
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = 1
        )
        self.point_conv_3 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels * 2,
            kernel_size = 1
        )
        self.point_conv_4 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = 1
        )
        
        self.dw_conv = nn.Conv2d(
            in_channels = in_channels * 2,
            out_channels = in_channels * 2,
            kernel_size = 1,    # Kernel size for DW Conv not specified in Paper.
            padding = "same",   # If we change kernel size, will pad input so that output shape remains the same.
            groups = in_channels * 2    # DW Conv has each input channel convolved separately
        )
        
        # Not much information as to how to implement Gate Block in Paper.
        # Assume a GLU is correct, but paper mentions not using any activation functions,
        # which are a part of GLU, so I'm not 100% sure.
        self.gate_1 = nn.GLU(dim = 1)
        self.gate_2 = nn.GLU(dim = 1)
        
        self.attention = ChannelAttention(in_channels = in_channels, reduction_ratio = 1)
        
        self.glfb_layer_1 = nn.Sequential(
            self.norm_1,
            self.point_conv_1,
            self.dw_conv,
            self.gate_1,
            self.attention,
            self.point_conv_2
        )
        
        self.glfb_layer_2 = nn.Sequential(
            self.norm_2,
            self.point_conv_3,
            self.gate_2,
            self.point_conv_4
        )
        
        
    def forward(self, x):
        l1 = self.glfb_layer_1(x)
        y = torch.add(l1, x)
        l2 = self.glfb_layer_2(y)
        out = torch.add(l2, y)
        return out
        
    
    
class TestGLFB(unittest.TestCase):
    def test_glfb_shape(self):
        batch_size, in_channels, features, signals = 3, 2, 4, 4
        input = torch.rand((batch_size, in_channels, features, signals))
        model = GLFB(in_channels, features, signals)
        output = model.forward(input)
        N, C, F, S = output.shape
        self.assertTrue(
            (N == batch_size) 
            and (C == in_channels) 
            and (F == features) 
            and (S == signals)
        )

        
        
if __name__ == "__main__":
    unittest.main()