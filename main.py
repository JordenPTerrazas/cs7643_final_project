import yaml
import argparse
import time
import copy
import unittest

import numpy as np
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
import torchaudio.transforms as transforms
import torchaudio.functional as F

from transforms.not_our_stdct import sdct_torch, isdct_torch
from modules import MFNet
from losses import TotalLoss
from dataloader import DNSDataset

parser = argparse.ArgumentParser(description='CS7643 Final Project')
parser.add_argument('--config', default='./configs/config.yaml')

window_eps = 1e-8

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def PESQ(output, target, mode = 'wb'):
    """
    Calculates the Perceptual Evaluation of Speech Quality metric.
    
    PESQ is recognized industry standard for audio quality that takes into considerations characteristics such as: 
    audio sharpness, call volume, background noise, clipping, audio interference etc. 
    PESQ returns a score between -0.5 and 4.5 with the higher scores indicating a better quality.
    
    This implementation uses the torchmetrics library from Lightning AI described here: 
    https://lightning.ai/docs/torchmetrics/stable/audio/perceptual_evaluation_speech_quality.html
    """
    if mode == 'nb':
        fs = 8000
    elif mode == 'wb':
        fs = 16000
    pesq = PerceptualEvaluationSpeechQuality(fs, 'wb')
    return pesq(output, target)


def STOI(output, target):
    """
    Calculate Short-Time Objective Intelligibility metric for evaluating speech signals.
    
    STOI is highly correlated with the intelligibility of degraded speech signals.
    
    This implementation uses the torchmetrics library from Lightning AI described here: 
    https://lightning.ai/docs/torchmetrics/stable/audio/short_time_objective_intelligibility.html
    """
    stoi = ShortTimeObjectiveIntelligibility(16000, False)
    return stoi(output, target)


def adjust_learning_rate(optimizer, epoch, args):
    epoch += 1
    if epoch > args.steps[1]:
        lr = args.lr * 0.01
    elif epoch > args.steps[0]:
        lr = args.lr * 0.1
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

def train(epoch, data_loader, model, optimizer, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    pesq = AverageMeter()
    stoi = AverageMeter()
    
    if torch.cuda.is_available():
        model.to("cuda")
   
    for idx, (input_waveforms, target_waveforms) in enumerate(data_loader):
        start = time.time()
        
        # FOR TESTING PURPOSES
        # TO VERIFY THAT MODEL IS LEARNING
        # target, sample_rate = torchaudio.load("./data/datasets/DNS_subset_10/clean/clean_fileid_0.wav")
        # data, sample_rate = torchaudio.load("./data/datasets/DNS_subset_10/noisy/book_11284_chp_0013_reader_05262_6_59oHl43FnXw_snr8_fileid_0.wav")
        # data, target = data[None,:,:], target[None,:,:]
        # print(data.shape, target.shape)
        # # # # # # # # # # # #
        # # # # # # # # # # # #
        
        if torch.cuda.is_available():
            input_waveforms = input_waveforms.cuda()
            target_waveforms = target_waveforms.cuda()
        
        # Fwd pass 
        out = model.forward(input_waveforms)
        
        # Remove padding (9 comes from 16 - 999 % 16, e.g. we can code to be dynamic later)
        out = out[:,:,:,:-9]
        
        # Compute loss then backwards
        loss = criterion(out, target_waveforms)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_stoi = STOI(out, target_waveforms)
        # Inverse Transform
        i_out = isdct_torch(out, frame_step=160, frame_length=320, window=torch.sqrt(torch.hann_window(window_length=320)).cuda() + window_eps) 
        i_target = isdct_torch(target_waveforms, frame_step=160, frame_length=320, window=torch.sqrt(torch.hann_window(window_length=320)).cuda() + window_eps)
        
        # Clean up large values from Inverse Transform
        i_target = torch.nan_to_num(i_target, nan=0.0, posinf=40.0, neginf=-40.0)
        i_out = torch.nan_to_num(i_out, nan=0.0, posinf=40.0, neginf=-40.0)
        print("Max in target", torch.max(i_target))
        print("Max in out", torch.max(i_out))

        # Compute PESQ & STOI 
        batch_pesq = PESQ(i_out, i_target)
        #batch_stoi = STOI(i_out, i_target)

        # Update Everything
        batch_size = out.shape[0]
        losses.update(loss.item(), out.shape[0])
        pesq.update(batch_pesq, batch_size)
        stoi.update(batch_stoi, batch_size)
        iter_time.update(time.time() - start)
        
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'PESQ {pesq.val:.4f} ({pesq.avg:.4f})\t'
                   'STOI {stoi.val:.4f} ({stoi.avg:.4f})\t').format(
                       epoch,
                       idx,
                       len(data_loader),
                       iter_time=iter_time,
                       loss=losses,
                       pesq=pesq,
                       stoi=stoi))


def validate(epoch, val_loader, model, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    pesq = AverageMeter()
    stoi = AverageMeter()

    for idx, (data, target) in enumerate(val_loader):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        with torch.no_grad():
            out = model(data)
            out = out[:,:,:,:-9]
            loss = criterion(out, target)

        i_out = isdct_torch(out, frame_step=160, frame_length=320, window=torch.sqrt(torch.hann_window(window_length=320)).cuda() + window_eps)
        i_target = isdct_torch(target, frame_step=160, frame_length=320, window=torch.sqrt(torch.hann_window(window_length=320)).cuda() + window_eps)
        
        # Clean up large values from Inverse Transform
        i_target = torch.nan_to_num(i_target, nan=0.0, posinf=40.0, neginf=-40.0)
        i_out = torch.nan_to_num(i_out, nan=0.0, posinf=40.0, neginf=-40.0)
        
        batch_pesq = PESQ(i_out, i_target)
        batch_stoi = STOI(i_out, i_target)
        
        batch_size = out.shape[0]
        losses.update(loss.item(), out.shape[0])
        pesq.update(batch_pesq, batch_size)
        stoi.update(batch_stoi, batch_size)
        iter_time.update(time.time() - start)
        
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'PESQ {pesq.val:.4f} ({pesq.avg:.4f})\t'
                   'STOI {stoi.val:.4f} ({stoi.avg:.4f})\t').format(
                       epoch,
                       idx,
                       len(val_loader),
                       iter_time=iter_time,
                       loss=losses,
                       pesq=pesq,
                       stoi=stoi))

    print(('* PESQ: {pesq.avg:.4f}\t* STOI: {stoi.avg:.4f}\t').format(pesq=pesq, stoi=stoi))
    return pesq.avg, stoi.avg


def main():
    # waveform, sample_rate = torchaudio.load("data/datasets/blind_test_set/noreverb_fileid_0.wav")
    # stdct_waveform = sdct_torch(waveform, 320, 160, torch.hann_window)
    
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    # train_dataset = torchaudio.datasets.LIBRISPEECH(
    #     root = "./data/datasets/LIBRISPEECH",
    #     url = "dev-clean",
    #     download = True
    # )
    # train_loader = DataLoader(
    #     train_dataset, 
    #     batch_size=args.batch_size, 
    #     shuffle=True
    # )
    # test_dataset = None
    # test_loader = None
    train_dataset = DNSDataset(
        clean_dir=args.train_label_dir,
        noisy_dir=args.train_data_dir,
        transform=sdct_torch,
        transform_kwargs={"frame_length": 320, "frame_step": 160, "window": torch.sqrt(torch.hann_window(window_length=320)) + window_eps}
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_dataset = DNSDataset(
        clean_dir=args.val_label_dir,
        noisy_dir=args.val_data_dir,
        transform=sdct_torch,
        transform_kwargs={"frame_length": 320, "frame_step": 160, "window": torch.sqrt(torch.hann_window(window_length=320)) + window_eps}
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    # Hyper-Parameters: gamma, lr, betas, weight_decay, epochs
    if args.model == "MFNet":
        model = MFNet(in_channels = 1, out_channels = 16)
    else:   # Could place modified model here
        pass

    criterion = TotalLoss(gamma = 0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = args.lr,
        betas = args.betas,
        weight_decay = args.weight_decay
    )
    
    best_pesq = 0.0
    best_stoi = 0.0
    best_model = None
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train loop
        train(epoch, train_dataloader, model, optimizer, criterion)

        # validation loop
        pesq, stoi = validate(epoch, val_dataloader, model, criterion)

        if pesq > best_pesq and stoi > best_stoi:
            best_pesq = pesq
            best_stoi = stoi
            best_model = copy.deepcopy(model)

    print('Best Prec @1 PESQ: {:.4f}'.format(pesq))
    print('Best Prec @1 STOI: {:.4f}'.format(stoi))

    if args.save_best:
        torch.save(best_model.state_dict(), './checkpoints/' + args.model + '.pth')


class TestMain(unittest.TestCase):
    def test_mfnet(self):
        train_dataset = torchaudio.datasets.LIBRISPEECH(
            root = "./data/datasets/LIBRISPEECH",
            url = "dev-clean",
            download = True
        )
        
        # Load the data
        waveform, sample_rate = torchaudio.load("./data/datasets/blind_test_set/noreverb_fileid_0.wav")
        noise, _ = torchaudio.load("./data/datasets/blind_test_set/noreverb_fileid_1.wav")
        snr_dbs = torch.tensor([20, 10, 3])
        noisy_speeches = F.add_noise(waveform, noise, snr_dbs)
        print(waveform.shape, sample_rate)
        print(noisy_speeches.shape)

        self.assertTrue(True)

if __name__ == '__main__':
    main()
