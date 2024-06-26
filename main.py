import os
import yaml
import argparse
import time
import copy
import unittest

import numpy as np
import torch
import torchaudio
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
import torchaudio.transforms as transforms
import torchaudio.functional as F
from torch.utils.tensorboard import SummaryWriter

from transforms.not_our_stdct import sdct_torch, isdct_torch
from modules import MFNet, MFNetAct, MFNetNoSigmoid
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

    try:
        out = pesq(output, target)
    except:
        print("Error in PESQ")
        out = torch.tensor(-0.5)
    return out


def STOI(output, target):
    """
    Calculate Short-Time Objective Intelligibility metric for evaluating speech signals.
    
    STOI is highly correlated with the intelligibility of degraded speech signals.
    
    This implementation uses the torchmetrics library from Lightning AI described here: 
    https://lightning.ai/docs/torchmetrics/stable/audio/short_time_objective_intelligibility.html
    """
    stoi = ShortTimeObjectiveIntelligibility(16000, False)
    return stoi(output, target)

def train(epoch, data_loader, model, optimizer, criterion, writer, scheduler):
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

        # Before the step, log gradients
        for name, param in model.named_parameters():
            writer.add_histogram(name + '/grad', param.grad, epoch * len(data_loader) + idx)

        optimizer.step()
        
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
        batch_stoi = STOI(i_out, i_target)

        # Update Everything
        batch_size = out.shape[0]
        losses.update(loss.item(), out.shape[0])
        pesq.update(batch_pesq, batch_size)
        stoi.update(batch_stoi, batch_size)
        iter_time.update(time.time() - start)

        if idx % args.save_every == 0:
            save_dir = args.save_dir + '/checkpoints'
            if os.path.exists(save_dir):
                torch.save(model.state_dict(), save_dir + '/' + args.model + f'_epoch{epoch}' + f'_step{idx}' + '.pth')
                torch.save(optimizer.state_dict(), save_dir + '/' + 'optim' + f'_epoch{epoch}' + f'_step{idx}' + '_optimizer.pth')
                torch.save(scheduler.state_dict(), save_dir + '/' + 'scheduler' + f'_epoch{epoch}' + f'_step{idx}' + '_scheduler.pth')
            else:
                os.makedirs(save_dir)
                torch.save(model.state_dict(), save_dir + '/' + args.model + f'_epoch{epoch}' + f'_step{idx}' + '.pth')
                torch.save(optimizer.state_dict(), save_dir + '/' + 'optim' + f'_epoch{epoch}' + f'_step{idx}' + '_optimizer.pth')
                torch.save(scheduler.state_dict(), save_dir + '/' + 'scheduler' + f'_epoch{epoch}' + f'_step{idx}' + '_scheduler.pth')
        
        if idx % 10 == 0:
            writer.add_scalar('Loss/train', loss.item(), epoch * len(data_loader) + idx)
            writer.add_scalar('PESQ/train', batch_pesq, epoch * len(data_loader) + idx)
            writer.add_scalar('STOI/train', batch_stoi, epoch * len(data_loader) + idx)
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


def validate(epoch, val_loader, model, criterion, writer):
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
            writer.add_scalar('Loss/val', loss.item(), epoch * len(val_loader) + idx)
            writer.add_scalar('PESQ/val', batch_pesq, epoch * len(val_loader) + idx)
            writer.add_scalar('STOI/val', batch_stoi, epoch * len(val_loader) + idx)
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
    start_epoch = 0
    if args.model == "MFNet":
        model = MFNet(in_channels = 1, out_channels = 16)

        if args.load_checkpoint:
            model.load_state_dict(torch.load(args.checkpoint_model_path))
            start_epoch = args.start_epoch
        else:
            start_epoch = 0
    elif args.model == "MFNetAct":
        model = MFNetAct(in_channels = 1, out_channels = 16)
        
        if args.load_checkpoint:
            model.load_state_dict(torch.load(args.checkpoint_model_path))
            start_epoch = args.start_epoch
        else:
            start_epoch = 0
    elif args.model == "MFNetNoSigmoid":
        model = MFNetNoSigmoid(in_channels = 1, out_channels = 16)

        if args.load_checkpoint:
            model.load_state_dict(torch.load(args.checkpoint_model_path))
            start_epoch = args.start_epoch
        else:
            start_epoch = 0
    else:
        pass

    criterion = TotalLoss(gamma = 0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = args.lr,
        betas = args.betas,
        weight_decay = args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.anneal_epochs, eta_min=0.0001)

    if args.load_checkpoint:
        optimizer.load_state_dict(torch.load(args.checkpoint_optimizer_path))
        scheduler.load_state_dict(torch.load(args.checkpoint_scheduler_path))
    
    if os.path.exists(args.save_dir + '/logs'):
        writer = SummaryWriter(args.save_dir + '/logs')
    else:
        os.makedirs(args.save_dir + '/logs')
        writer = SummaryWriter(args.save_dir + '/logs')

    best_pesq = 0.0
    best_stoi = 0.0
    best_model = None
    for epoch in range(start_epoch, args.epochs):
        # train loop
        train(epoch, train_dataloader, model, optimizer, criterion, writer, scheduler)

        scheduler.step()

        # validation loop
        pesq, stoi = validate(epoch, val_dataloader, model, criterion, writer)

        if pesq > best_pesq and stoi > best_stoi:
            best_pesq = pesq
            best_stoi = stoi
            best_model = copy.deepcopy(model)

            if args.save_best:
                best_save_dir = args.save_dir + '/best'
                if os.path.exists(best_save_dir):
                    torch.save(best_model.state_dict(), best_save_dir + '/' + args.model + '.pth')
                else:
                    os.makedirs(best_save_dir)
                    torch.save(best_model.state_dict(), best_save_dir + '/' + args.model + '.pth')

            print('Best Prec @1 PESQ: {:.4f}'.format(best_pesq))
            print('Best Prec @1 STOI: {:.4f}'.format(best_stoi))

    writer.close()
    
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
