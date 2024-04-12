import torch
import torchaudio
import torch.nn as nn
import unittest

from .AbsoluteLoss import AbsoluteLoss
from .PolarLoss import PolarLoss


class TotalLoss(nn.Module):
    """
    Total Loss is implemented as discussed in Section 2.5 of https://arxiv.org/pdf/2306.04286.pdf.
    The absolute and polar losses are combined using gamma to weight their relative impact.
    """
    
    def __init__(self, gamma = 0.5):
        super(TotalLoss, self).__init__()
        assert gamma >= 0 and gamma <= 1
        self.gamma = gamma
        self.absolute_loss = AbsoluteLoss()
        self.polar_loss = PolarLoss()
        
    def forward(self, input, target):
        loss = torch.add(
            torch.mul(self.absolute_loss(input, target), self.gamma),
            torch.mul(self.polar_loss(input, target), (1 - self.gamma))
        )
        return loss
    
    
class TestTotalLoss(unittest.TestCase):
    def test_total_loss(self):
        input = torch.Tensor([1, -2, 4])
        target = torch.Tensor([1, 2, 3])
        gamma = 0.5
        model = TotalLoss(gamma = gamma)
        loss = model.forward(input, target)
        self.assertTrue(
            loss == 
                gamma * (((1 - 1) ** 2 + (2 - 2) ** 2 + (3 - 4) ** 2) / 3)
                + (1 - gamma) * (((1 - 1) ** 2 + (2 - -2) ** 2 + (3 - 4) ** 2) / 3)
        )
        

if __name__ == '__main__':
    unittest.main()