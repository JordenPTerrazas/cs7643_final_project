import torch
import torchaudio
import torch.nn as nn
import unittest

class GLFB(nn.Module):
    """

    """
    
    def __init__(self, in_channels):
        super(GLFB, self).__init__()
        self.glfb_layer_1 = nn.Sequential(
            # Layer Norm
            nn.Conv2d(  # Point Conv 1 & 3 double number of channels (https://arxiv.org/pdf/2306.04286.pdf Section 2.4)
                in_channels = in_channels,
                out_channels = in_channels * 2,
                kernel_size = 1
            ),
            # Depth Wise Seperable Convolution
            # Gate
            # Channel Attention
            nn.Conv2d(  # Point Conv 2 & 4 halve number of channels (https://arxiv.org/pdf/2306.04286.pdf Section 2.4)
                in_channels = in_channels * 2,
                out_channels = in_channels,
                kernel_size = 1
            )
        )
        self.glfb_layer_2 = nn.Sequential(
            # Layer Norm
            nn.Conv2d(  # Point Conv 1 & 3 double number of channels (https://arxiv.org/pdf/2306.04286.pdf Section 2.4)
                in_channels = in_channels,
                out_channels = in_channels * 2,
                kernel_size = 1
            ),
            # Gate
            nn.Conv2d(  # Point Conv 2 & 4 halve number of channels (https://arxiv.org/pdf/2306.04286.pdf Section 2.4)
                in_channels = in_channels * 2,
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
        pass

        
        
if __name__ == "__main__":
    unittest.main()