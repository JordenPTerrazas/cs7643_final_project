"""Here I try to reimplement jonashaag & zh217's DCT implementation and compare it to our own version of the 
short-time discrete cosine transformation from 'Real-Time Monaural Speech Enhancement With Short-time Discrete
Cosine Transform' Li et, al."""
import os
import time
import unittest
import torch
import torchaudio
import numpy as np
from .not_our_stdct import sdct_torch

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
        stdct_of_waveform (Tensor): Tensor of shape [channels, window_size, ((signal_samples-window_size)/hop_size) + 1] 
        representing the DCT of the waveform.
    """
    # Helper function for coefficient on the sum
    def c_mu(index):
        if index == 0:
            return np.sqrt(1.0 / 2.0)
        else:
            return 1.0
    
    # Constants
    N = waveform.size(1)
    window_size = window.size(0)

    # Compute the DCT of the waveform
    stdct_of_waveform = torch.zeros_like(waveform)
    for u in range(0, N - window_size + 1, hop_size):
        for n in range(0, N - window_size + 1, hop_size):
            # This is the sum in the DCT formula from the paper, with windowing added
            stdct_of_waveform[:, u:u+window_size] += c_mu(u) * np.sqrt(2.0 / N) * waveform[:, n:n+window_size] * window * np.cos(np.pi * u * (2 * n + 1) / (2 * N))

    return stdct_of_waveform.unfold(-1, window_size, hop_size).transpose(-1, -2)

def fast_stdct(waveform, window, hop_size):
    """
    Our FAST reimplementation of jonashaag & zh217's short-time discrete cosine transformation 
    Takes a waveform tensor as input and returns the DCT of the waveform.

    Args:
        waveform (Tensor): Tensor of shape [channels, signal_samples] representing the waveform.
        window (Tensor): Tensor of shape [window_size] representing the window function to apply to the waveform.
        hop_size (int): The number of signal samples to move the window by each iteration.
    
    Returns:
        stdct_of_waveform (Tensor): Tensor of shape [channels, window_size, ((signal_samples-window_size)/hop_size) + 1] 
        representing the DCT of the waveform.
    """
    # Save Sizes
    N = waveform.size(1)
    window_size = window.size(0)

    # Apply framing based on window and hop size
    framed_signal = waveform.unfold(-1, window_size, hop_size)
    
    # Save sizes
    Framed_N_size = framed_signal.size()
    Framed_N = Framed_N_size[-1]
    
    # Apply the window
    framed_signal = framed_signal * window

    # Convert to a 2D tensor, ignoring channel
    framed_signal = framed_signal.view(-1, Framed_N)

    # Double the signal with the second half reversed
    doubled_signal = torch.cat(
        [framed_signal[:, ::2] , framed_signal[:, 1::2 ]], dim = 1
    )

    # Apply FFT and take the real part
    real_part = torch.view_as_real(torch.fft.fft(doubled_signal, dim = 1))

    # To get the actual DCT coefficients we have to phase shift the real part a bit
    k = -torch.arange(
        Framed_N, dtype = framed_signal.dtype, device = framed_signal.device
        ) * np.pi / (2 * Framed_N)
    
    real_coef = torch.cos(k)
    img_coef = torch.sin(k)

    shifted_transform = real_coef * real_part[:, :, 0] - img_coef * real_part[:, :, 1]
    transformed_signal = 2 * shifted_transform.view(*Framed_N_size)

    return transformed_signal.transpose(-1, -2)

class TestSTDCT(unittest.TestCase):
    def test_stdct(self):
        waveform, sample_rate = torchaudio.load("data/MFNet_enhanced_DNS/fileid_0.wav")
        window = torch.hann_window(
            window_length = 320
        )
        ot_0 = time.time()
        our_stdct_of_waveform = stdct(waveform, window, 160)
        print("our shape", our_stdct_of_waveform.size())
        ot_1 = time.time()
        print("Our STDCT took", ot_1 - ot_0, "seconds")

        nt_0 = time.time()
        not_our_stdct_of_waveform = sdct_torch(waveform, 320, 160, torch.hann_window)
        nt_1 = time.time()
        print("Not our STDCT took", nt_1 - nt_0, "seconds")

        oft_0 = time.time()
        our_fast_stdct_of_waveform = fast_stdct(waveform, window, 160)
        oft_1 = time.time()
        print("Our FAST STDCT took", oft_1 - oft_0, "seconds")

        print("Diff Ours - Fast: ", torch.sum(our_stdct_of_waveform).item() - torch.sum(not_our_stdct_of_waveform).item())
        print("Diff Our Fast - Fast: ", torch.sum(our_stdct_of_waveform).item() - torch.sum(not_our_stdct_of_waveform).item())
        print("Diff Ours - Our Fast: ", torch.sum(our_stdct_of_waveform).item() - torch.sum(our_fast_stdct_of_waveform).item())
        self.assertTrue(np.abs(torch.sum(our_fast_stdct_of_waveform).item() - torch.sum(our_stdct_of_waveform).item()) < 2)
        # So the differences betwen our fast / their fast have to do with normalization, and the differences between
        # our slow / their fast have to do with normalization + windowing. Since Li et, al dont mention the normalization but
        # it is common practice to normalize spectrograms, we'll suppose they use ortho normalization and use jonashaag & zh217's
        # implementation

if __name__ == '__main__':
    unittest.main()
