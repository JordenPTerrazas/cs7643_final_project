"""Here I try to implement our own version of the short-time discrete cosine transformation, compare it to existing
implementations as a test, and then try to implement a fast version of the DCT using PyTorch's FFT implementation, motivated
by zh217's implementation of the DCT here https://github.com/zh217/torch-dct"""
import os
import time
import unittest
import torch
import torchaudio
import numpy as np
from not_our_stdct import sdct_torch

def stdct(waveform, window, hop_size):
    """
    Our implementation of the short-time discrete cosine transformation as described
    in 'Real-Time Monaural Speech Enhancement With Short-time Discrete Cosine Transform'
    Li et, al. Takes a waveform tensor as input and returns the DCT of the waveform.

    Args:
        waveform (Tensor): Tensor of shape [channels, signal_samples] representing the waveform.
        window (Tensor): Tensor of shape [window_size] representing the window function to apply to the waveform.
        hop_size (int): The number of signal samples to move the window by each iteration.
    
    Returns:
        stdct_of_waveform (Tensor): Tensor of shape [channels, signal_samples] representing the DCT of the waveform.
    """
    # Helper function for coefficient on the sum
    def c_mu(index):
        if index == 0:
            return np.sqrt(1.0 / 2.0)
        else:
            return 1.0

    N = waveform.size(1)
    window_size = window.size(0)
    
    # Compute the DCT of the waveform
    stdct_of_waveform = torch.zeros_like(waveform)
    for u in range(0, N - window_size + 1, hop_size):
        for n in range(0, N - window_size + 1, hop_size):
            stdct_of_waveform[:, u:u+window_size] += c_mu(u) * np.sqrt(2.0 / N) * waveform[:, n:n+window_size] * window * np.cos(np.pi * u * (2 * n + 1) / (2 * N))

    return stdct_of_waveform.unfold(-1, window_size, hop_size).transpose(-1, -2)

class TestSTDCT(unittest.TestCase):
    def test_stdct(self):
        waveform, sample_rate = torchaudio.load("data/MFNet_enhanced_DNS/fileid_0.wav")
        window = torch.hann_window(
            window_length = 320
        )
        ot_0 = time.time()
        our_stdct_of_waveform = stdct(waveform, window, 160)
        ot_1 = time.time()
        print("Our STDCT took", ot_1 - ot_0, "seconds")

        nt_0 = time.time()
        not_our_stdct_of_waveform = sdct_torch(waveform, 320, 160, torch.hann_window)
        nt_1 = time.time()
        print("Not our STDCT took", nt_1 - nt_0, "seconds")

        print("Diff: ", torch.sum(our_stdct_of_waveform - not_our_stdct_of_waveform).item())
        self.assertTrue(torch.allclose(our_stdct_of_waveform, not_our_stdct_of_waveform, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
