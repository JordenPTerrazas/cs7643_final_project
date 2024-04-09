import torch
import torchaudio
import torch.nn as nn
import unittest

class UpSample(nn.Module):
    """
    The UpSample Block doubles the features and signals and halves the number of channels.
    PixelShuffle is used to avoid the checkerboard grid effect that can occur with
    transposed convolution, as discussed in section 2.3 of https://arxiv.org/pdf/2306.04286.pdf
    """
    
    def __init__(self, in_channels):
        super(UpSample, self).__init__()
        
        self.up_sample_layer = nn.Sequential(
            nn.PixelShuffle(2), # Space => features * 2 X signals * 2, Depth => in_channels // 4
            nn.Conv2d(  # Depth => in_channels // 2
                in_channels = in_channels // 4, 
                out_channels = in_channels // 2, 
                kernel_size = 1
            )   
        )
        
    def forward(self, x):
        """
        Args:
            x (Tensor): [batch_size, in_channels, features, signals]

        Returns:
            out (Tensor): [batch_size, in_channels // 2, features * 2, signals * 2]
        """
        
        out = self.up_sample_layer(x)
        return out
    
    
class TestUpSample(unittest.TestCase):
    def test_up_sample_shape(self):
        batch_size, features, signals, in_channels = 2, 7, 8, 8
        input = torch.zeros((batch_size, in_channels, features, signals))
        model = UpSample(in_channels)
        output = model.forward(input)
        N, C, F, S = output.shape
        print("Input Shape: ", input.shape)
        print("Output Shape: ", output.shape)
        self.assertTrue((N == batch_size) and (C == in_channels // 2) and (F == features * 2) and (S == signals * 2))

        
        
if __name__ == "__main__":
    unittest.main()