import torch
import torchaudio
import torch.nn as nn
import unittest


class PolarLoss(nn.Module):
    """
    Mean Squared Error Loss for the polar values of the input and the target.
    Implemented as discussed in Section 2.5 of https://arxiv.org/pdf/2306.04286.pdf
    """
    
    def __init__(self):
        super(PolarLoss, self).__init__()
        self.loss = nn.MSELoss(reduction = "mean")
        
    def forward(self, input, target):
        output = self.loss(input, target)
        return output
    
    
class TestPolarLoss(unittest.TestCase):
    def test_polar_loss(self):
        pass

if __name__ == '__main__':
    unittest.main()