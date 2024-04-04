import torch
import torchaudio
import torch.nn as nn
import unittest

class DownSample(nn.Module):
    """
    The Down Sample Block halves the number of features and signals
    and doubles the number of channels
    """
    
    def __init__(self, in_channels):
        super(DownSample, self).__init__()
        self.down_sample_layer = nn.Conv2d(
            in_channels = in_channels, 
            out_channels = in_channels * 2,
            kernel_size = (2,2),    # Specified in https://arxiv.org/pdf/2306.04286.pdf in Section 2.3
            stride = 2,
            padding = 0
        )
        
    def forward(self, x):
        """
        Args:
            x (Tensor): [batch_size, in_channels, features, signal]

        Returns:
            out (Tensor): [batch_size, in_channels * 2, features // 2, signal // 2]
        """
        
        out = self.down_sample_layer(x)
        return out
    
    
class TestDownSample(unittest.TestCase):
    def test_down_sample_shape(self):
        batch_size, features, signals, in_channels = 2, 7, 8, 8
        input = torch.zeros((batch_size, in_channels, features, signals))
        model = DownSample(in_channels)
        output = model.forward(input)
        N, C, F, S = output.shape
        self.assertTrue((N == batch_size) and (C == in_channels * 2) and (F == features // 2) and (S == signals // 2))

        
        
if __name__ == "__main__":
    unittest.main()