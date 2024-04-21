import unittest
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
#from transforms.not_our_stdct import sdct_torch

# class DNSDataset(Dataset):
#     def __init__(self, directory, transform=None, transform_kwargs={}):
#         self.directory = directory
#         self.transform = transform
#         self.transform_kwargs = transform_kwargs
#         self.files = [os.path.join(directory, file) for file in os.listdir(directory)]

#     def __len__(self):
#         return len(self.files)
    
#     def __getitem__(self, idx):
#         audio_file = self.files[idx]
#         waveform, sample_rate = torchaudio.load(audio_file)

#         if self.transform:
#             waveform = self.transform(waveform, **self.transform_kwargs)

#         return waveform, sample_rate
    
class DNSDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, transform=None, transform_kwargs={}):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.transform = transform
        self.transform_kwargs = transform_kwargs
        self.clean_files = [os.path.join(clean_dir, file) for file in os.listdir(clean_dir)]
        self.noisy_files = [os.path.join(noisy_dir, file) for file in os.listdir(noisy_dir)]

    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        noisy_file = self.noisy_files[idx]
        clean_file = os.path.join(self.clean_dir, f"clean_fileid_{noisy_file.split('fileid_')[-1].split('.')[0]}.wav")

        noisy_waveform, sample_rate = torchaudio.load(noisy_file)
        clean_waveform, sample_rate = torchaudio.load(clean_file)

        if self.transform:
            noisy_waveform = self.transform(noisy_waveform, **self.transform_kwargs)
            clean_waveform = self.transform(clean_waveform, **self.transform_kwargs)

        return noisy_waveform, clean_waveform
        
    
class TestLoading(unittest.TestCase):
    def test_dns_load(self):
        dataset = DNSDataset(
            "data/datasets/DNS_subset_10/train/clean", 
            "data/datasets/DNS_subset_10/train/noisy",
            transform=sdct_torch,
            transform_kwargs={"frame_length": 320, "frame_step": 160, "window": torch.hann_window}
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        for idx, batch in enumerate(dataloader):
            if idx > 1:
                break
            noisy_waveform, clean_waveform = batch
            print("noisy waveform shape: ", noisy_waveform.shape)
            print("clean waveform shape: ", clean_waveform.shape)


if __name__ == "__main__":
    unittest.main()