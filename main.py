import yaml
import argparse
import time
import copy

import numpy as np
import torch
import torchaudio

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio.transforms as transforms

from transforms.stdct import fast_stdct
from modules import MFNet
from losses import TotalLoss


def main():
    print(torchaudio.info("datasets/blind_test_set/noreverb_fileid_0.wav"))


if __name__ == '__main__':
    main()
