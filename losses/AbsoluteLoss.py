import torch
import torchaudio
import torch.nn as nn
import unittest


class AbsoluteLoss(nn.Module):
    """
    Mean Squared Error Loss for the absolute values of the input and the target.
    Implemented as discussed in Section 2.5 of https://arxiv.org/pdf/2306.04286.pdf
    """
    
    def __init__(self):
        super(AbsoluteLoss, self).__init__()
        self.loss = nn.MSELoss(reduction = "mean")
        
    def forward(self, input, target):
        abs_input, abs_target = torch.abs(input), torch.abs(target)
        output = self.loss(abs_input, abs_target)
        return output
    
    
class TestAbsoluteLoss(unittest.TestCase):
    def test_absolute_loss(self):
        pass

if __name__ == '__main__':
    unittest.main()
        