import torch
import torchaudio
import torch.nn as nn
import unittest

class Projection(nn.Module):
    """
    The Projection Layer should maintain the size of the input feature map
    and increase the number of channels from 1 to n.
    """
    
    def __init__(self, in_channels, out_channels):
        super(Projection, self).__init__()
        self.proj_layer = nn.Conv2d(
            in_channels = in_channels, 
            out_channels = out_channels,
            kernel_size = (3,3),    # Specified in https://arxiv.org/pdf/2306.04286.pdf in Section 2.3
            padding = "same"
        )
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Tensor [batch_size, in_channels, features, signal_samples] representing waveforms.

        Returns:
            out (Tensor): Tensor [batch_size, out_channels, features, signal_samples] of projected waveforms.
        """
        
        out = self.proj_layer(x)
        return out
    
    
class TestProjection(unittest.TestCase):
    def test_proj_shape(self):
        batch_size, features, signals, in_channels, out_channels = 2, 3, 4, 1, 5
        input = torch.zeros((batch_size, in_channels, features, signals))
        model = Projection(in_channels, out_channels)
        output = model.forward(input)
        N, C, F, S = output.shape
        self.assertTrue(N == batch_size and C == out_channels and F == features and S == signals)
        
        
if __name__ == "__main__":
    unittest.main()