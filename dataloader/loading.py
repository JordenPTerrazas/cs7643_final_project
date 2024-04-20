import unittest
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transforms.not_our_stdct import sdct_torch

class DNSDataset(Dataset):
    def __init__(self, directory, transform=None, transform_kwargs={}):
        self.directory = directory
        self.transform = transform
        self.transform_kwargs = transform_kwargs
        self.files = [os.path.join(directory, file) for file in os.listdir(directory)]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_file = self.files[idx]
        waveform, sample_rate = torchaudio.load(audio_file)

        if self.transform:
            waveform = self.transform(waveform, **self.transform_kwargs)

        return waveform, sample_rate
    
class TestLoading(unittest.TestCase):
    def test_dns_load(self):
        dataset = DNSDataset(
            "/home/jptau/cs7643_final_project/data/datasets/DNS_subset_10/clean", 
            transform=sdct_torch, 
            transform_kwargs={"frame_length": 320, "frame_step": 160, "window": torch.hann_window}
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        for idx, batch in enumerate(dataloader):
            if idx > 1:
                break
            waveform, sample_rate = batch
            print("waveform shape: ", waveform.shape)
            print("sample rate: ", sample_rate)

if __name__ == "__main__":
    unittest.main()