import os
import yaml
import argparse
import time
import torch
import torchaudio
from torch.utils.data import DataLoader
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torch.utils.tensorboard import SummaryWriter

from transforms.not_our_stdct import sdct_torch, isdct_torch
from modules import MFNet, MFNetAct, MFNetNoSigmoid
from losses import TotalLoss
from dataloader import DNSDataset, DNSTestSet

parser = argparse.ArgumentParser(description='CS7643 Final Project')
parser.add_argument('--config', default='./configs/infer_config.yaml')

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

def inference_synth(model, dataloader, criterion, name, writer):
    iter_time = AverageMeter()
    losses = AverageMeter()
    pesq = AverageMeter()
    stoi = AverageMeter()

    with torch.no_grad():
        for idx, (input_waveforms, target_waveforms) in enumerate(dataloader):
            start = time.time()
            
            if torch.cuda.is_available():
                input_waveforms = input_waveforms.cuda()
                target_waveforms = target_waveforms.cuda()
            
            # Forward pass
            out = model.forward(input_waveforms)

            # Remove padding
            out = out[:,:,:,:-9]

            # Calculate loss
            loss = criterion(out, target_waveforms)
            print(loss)

            # Inverse transform
            i_out = isdct_torch(out, frame_step=160, frame_length=320, window=torch.sqrt(torch.hann_window(window_length=320)).cuda() + window_eps)
            i_target = isdct_torch(target_waveforms, frame_step=160, frame_length=320, window=torch.sqrt(torch.hann_window(window_length=320)).cuda() + window_eps)

            # Catch numeric issues
            i_out = torch.nan_to_num(i_out, nan=0.0, posinf=40.0, neginf=-40.0)
            i_target = torch.nan_to_num(i_target, nan=0.0, posinf=40.0, neginf=-40.0)

            # Compute PESQ and STOI
            batch_pesq = PESQ(i_out, i_target)
            batch_stoi = STOI(i_out, i_target)

            # Update everything
            batch_size = out.shape[0]
            losses.update(loss.item(), batch_size)
            pesq.update(batch_pesq, batch_size)
            stoi.update(batch_stoi, batch_size)
            iter_time.update(time.time() - start)

            # Log to tensorboard
            writer.add_scalar(f"Loss/{name}", loss.item(), len(dataloader) + idx)
            writer.add_scalar(f"PESQ/{name}", batch_pesq, len(dataloader) + idx)
            writer.add_scalar(f"STOI/{name}", batch_stoi, len(dataloader) + idx)

            print(f"Batch {idx}: Loss {losses.val}, PESQ {pesq.val}, STOI {stoi.val}, Time {iter_time.val}")
        
        print(f"Results {name}: Loss {losses.avg}, PESQ {pesq.avg}, STOI {stoi.avg}, Time {iter_time.avg}")

def inference_real(model, dataloader):
    with torch.no_grad():
        for idx, input_waveforms in enumerate(dataloader):
            start = time.time()

            if torch.cuda.is_available():
                input_waveforms = input_waveforms.cuda()
            
            out = model.forward(input_waveforms)
            # Remove padding
            out = out[:,:,:,:-9] 

            # Inverse transform
            i_out = isdct_torch(out, frame_step=160, frame_length=320, window=torch.sqrt(torch.hann_window(window_length=320)).cuda() + window_eps)

            # Catch numeric issues
            i_out = torch.nan_to_num(i_out, nan=0.0, posinf=40.0, neginf=-40.0)

            if os.path.exists(args.real_save_dir):
                pass
            else:
                os.makedirs(args.real_save_dir)

            # Save the output
            for i in range(out.shape[0]):
                torchaudio.save(f"{args.real_save_dir}/real_output_{i}.wav", i_out[i].cpu(), 16000)

def main():
    # Load args
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    # Load the model
    if args.model == "MFNet":
        model = MFNet(in_channels = 1, out_channels = 16)
        model.load_state_dict(torch.load(args.weights))

    elif args.model == "MFNetAct":
        model = MFNetAct(in_channels = 1, out_channels = 16)
        model.load_state_dict(torch.load(args.weights))
    
    elif args.model == "MFNetNoSigmoid":
        model = MFNetNoSigmoid(in_channels = 1, out_channels = 16)
        model.load_state_dict(torch.load(args.weights))
    else:
        raise Exception("Invalid model name {}.".format(args.model))
    
    # Load data
    test_synth_no_reverb = DNSDataset(
        clean_dir = args.test_synth_noreverb_clean,
        noisy_dir = args.test_synth_noreverb_noisy,
        transform = sdct_torch,
        transform_kwargs = {
            "frame_length": 320,
            "frame_step": 160,
            "window" : torch.sqrt(torch.hann_window(window_length=320)) + window_eps
        }
    )

    test_synth_no_reverb_loader = DataLoader(
        test_synth_no_reverb,
        batch_size = args.batch_size,
        shuffle = True
    )

    test_synth_reverb = DNSDataset(
        clean_dir = args.test_synth_reverb_clean,
        noisy_dir = args.test_synth_reverb_noisy,
        transform = sdct_torch,
        transform_kwargs = {
            "frame_length": 320,
            "frame_step": 160,
            "window" : torch.sqrt(torch.hann_window(window_length=320)) + window_eps
        }
    )

    test_synth_reverb_loader = DataLoader(
        test_synth_reverb,
        batch_size = args.batch_size,
        shuffle = True
    )

    test_real = DNSTestSet(
        noisy_dir = args.test_real_noisy,
        transform = sdct_torch,
        transform_kwargs = {
            "frame_length": 320,
            "frame_step": 160,
            "window" : torch.sqrt(torch.hann_window(window_length=320)) + window_eps
        }
    )

    test_real_loader = DataLoader(
        test_real,
        batch_size = args.batch_size,
        shuffle = False
    )

    # Setup loss
    criterion = TotalLoss(gamma = 0.5)
    
    # Prep model for eval
    model.eval()
    model.to("cuda") if torch.cuda.is_available() else model.to("cpu")

    if args.real_only:
        inference_real(model, test_real_loader)
    else:
        writer = SummaryWriter()
        inference_synth(model, test_synth_no_reverb_loader, criterion, "no_rev", writer)
        inference_synth(model, test_synth_reverb_loader, criterion, "rev", writer)
        inference_real(model, test_real_loader)
        writer.close()

    return None

if __name__ == '__main__':
    main()