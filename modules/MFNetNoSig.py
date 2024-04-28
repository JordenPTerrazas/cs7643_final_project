import unittest
import torch
import torch.nn as nn
#from modules import GLFBNoSig, UpSample, DownSample, Projection
from .GLFBNoSigmoid import GLFBNoSig
from .UpSample import UpSample
from .DownSample import DownSample
from .Projection import Projection

# One thing not specified in the paper is the scenario where we end up with rounding
# error when we downsample and upsample. I'm going to assume that we should pad the
# result at the layer it occurs, but not sure on best practice.
class MFNetNoSigmoid(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 16):
        super(MFNetNoSigmoid, self).__init__()
        # Batch norm layers
        # self.bn_1 = nn.BatchNorm2d(out_channels)
        # self.bn_2 = nn.BatchNorm2d(2*out_channels)
        # self.bn_3 = nn.BatchNorm2d(4*out_channels)
        # self.bn_4 = nn.BatchNorm2d(8*out_channels)
        # self.bn_5 = nn.BatchNorm2d(8*out_channels)
        # self.bn_6 = nn.BatchNorm2d(4*out_channels)
        # self.bn_7 = nn.BatchNorm2d(2*out_channels)
        # self.bn_8 = nn.BatchNorm2d(out_channels)
        # self.bn_9 = nn.BatchNorm2d(in_channels)

        # Projection Layers (in_channels -> out_channels)
        self.projection_1 = Projection(
            in_channels = in_channels, 
            out_channels = out_channels
        )

        self.projection_2 = Projection(
            in_channels = out_channels,
            out_channels = in_channels
        )

        # GLFBNoSig Layers (out_channels -> out_channels)
        self.glfb_1 = GLFBNoSig(
            in_channels = out_channels,
            frame_size = 320,
            n_frames = 1008
        )

        self.glfb_2 = GLFBNoSig(
            in_channels = out_channels * 2,
            frame_size = 160,
            n_frames = 504
        )

        self.glfb_3_1 = GLFBNoSig(
            in_channels = out_channels * 4,
            frame_size = 80,
            n_frames = 252
        )

        self.glfb_3_2 = GLFBNoSig(
            in_channels = out_channels * 4,
            frame_size = 80,
            n_frames = 252
        )

        self.glfb_3_3 = GLFBNoSig(
            in_channels = out_channels * 4,
            frame_size = 80,
            n_frames = 252
        )

        self.glfb_3_4 = GLFBNoSig(
            in_channels = out_channels * 4,
            frame_size = 80,
            n_frames = 252
        )

        self.glfb_3_5 = GLFBNoSig(
            in_channels = out_channels * 4,
            frame_size = 80,
            n_frames = 252
        )

        self.glfb_3_6 = GLFBNoSig(
            in_channels = out_channels * 4,
            frame_size = 80,
            n_frames = 252
        )

        self.glfb_3_7 = GLFBNoSig(
            in_channels = out_channels * 4,
            frame_size = 80,
            n_frames = 252
        )

        self.glfb_3_8 = GLFBNoSig(
            in_channels = out_channels * 4,
            frame_size = 80,
            n_frames = 252
        )

        self.glfb_4_1 = GLFBNoSig(
            in_channels = out_channels * 8,
            frame_size = 40,
            n_frames = 126
        )

        self.glfb_4_2 = GLFBNoSig(
            in_channels = out_channels * 8,
            frame_size = 40,
            n_frames = 126
        )

        self.glfb_4_3 = GLFBNoSig(
            in_channels = out_channels * 8,
            frame_size = 40,
            n_frames = 126
        )

        self.glfb_4_4 = GLFBNoSig(
            in_channels = out_channels * 8,
            frame_size = 40,
            n_frames = 126
        )

        self.glfb_5_1 = GLFBNoSig(
            in_channels = out_channels * 16,
            frame_size = 20,
            n_frames = 63
        )

        self.glfb_5_2 = GLFBNoSig(
            in_channels = out_channels * 16,
            frame_size = 20,
            n_frames = 63
        )

        self.glfb_5_3 = GLFBNoSig(
            in_channels = out_channels * 16,
            frame_size = 20,
            n_frames = 63
        )

        self.glfb_5_4 = GLFBNoSig(
            in_channels = out_channels * 16,
            frame_size = 20,
            n_frames = 63
        )

        self.glfb_5_5 = GLFBNoSig(
            in_channels = out_channels * 16,
            frame_size = 20,
            n_frames = 63
        )

        self.glfb_5_6 = GLFBNoSig(
            in_channels = out_channels * 16,
            frame_size = 20,
            n_frames = 63
        )

        self.glfb_6 = GLFBNoSig(
            in_channels = out_channels * 8,
            frame_size = 40,
            n_frames = 126
        )

        self.glfb_7 = GLFBNoSig(
            in_channels = out_channels * 4,
            frame_size = 80,
            n_frames = 252
        )

        self.glfb_8 = GLFBNoSig(
            in_channels = out_channels * 2,
            frame_size = 160,
            n_frames = 504
        )

        self.glfb_9 = GLFBNoSig(
            in_channels = out_channels,
            frame_size = 320,
            n_frames = 1008
        )

        # Downsample Layers 
        self.downsample_1 = DownSample( # (out_channels -> out_channels * 2)
            in_channels = out_channels,
        )

        self.downsample_2 = DownSample( # (out_channels * 2 -> out_channels * 4)
            in_channels = out_channels * 2
        )

        self.downsample_3 = DownSample( # (out_channels * 4 -> out_channels * 8)
            in_channels = out_channels * 4
        )

        self.downsample_4 = DownSample( # (out_channels * 8 -> out_channels * 16)
            in_channels = out_channels * 8
        )

        # Upsample Layers
        self.upsample_1 = UpSample( # (out_channels * 16 -> out_channels * 8)
            in_channels = out_channels * 16
        )

        self.upsample_2 = UpSample( # (out_channels * 8 -> out_channels * 4)
            in_channels = out_channels * 8
        )

        self.upsample_3 = UpSample( # (out_channels * 4 -> out_channels * 2)
            in_channels = out_channels * 4
        )

        self.upsample_4 = UpSample( # (out_channels * 2 -> out_channels)
            in_channels = out_channels * 2
        )

    def forward(self, x):
        # Padding (advised directly by Liu Liang)
        x = nn.functional.pad(x, (0, 16 - x.shape[-1] % 16))

        # Encoder part
        x = self.projection_1(x)
        x1 = self.glfb_1(x)
        # BN
        # x1 = self.bn_1(x)

        x = self.downsample_1(x1)
        x2 = self.glfb_2(x)
        # BN
        # x2 = self.bn_2(x)

        x = self.downsample_2(x2)
        x = self.glfb_3_1(x)
        x = self.glfb_3_2(x)
        x = self.glfb_3_3(x)
        x = self.glfb_3_4(x)
        x = self.glfb_3_5(x)
        x = self.glfb_3_6(x)
        x = self.glfb_3_7(x)
        x3 = self.glfb_3_8(x)
        # BN
        # x3 = self.bn_3(x)

        x = self.downsample_3(x3)
        x = self.glfb_4_1(x)
        x = self.glfb_4_2(x)
        x = self.glfb_4_3(x)
        x4 = self.glfb_4_4(x)
        # BN
        # x4 = self.bn_4(x)

        # Bottleneck
        x = self.downsample_4(x4)
        x = self.glfb_5_1(x)
        x = self.glfb_5_2(x)
        x = self.glfb_5_3(x)
        x = self.glfb_5_4(x)
        x = self.glfb_5_5(x)
        x = self.glfb_5_6(x)
        x = self.upsample_1(x)
        x = torch.add(x, x4)
        # BN
        # x = self.bn_5(x)
        
        # Decoder part
        x = self.glfb_6(x)
        x = self.upsample_2(x)
        x = torch.add(x, x3)
        # BN
        # x = self.bn_6(x)

        x = self.glfb_7(x)
        x = self.upsample_3(x)
        x = torch.add(x, x2)
        # BN
        # x = self.bn_7(x)

        x = self.glfb_8(x)
        x = self.upsample_4(x)
        x = torch.add(x, x1)
        # BN
        # x = self.bn_8(x)

        x = self.glfb_9(x)
        x = self.projection_2(x)
        # x = self.bn_9(x)
        return x

class TestMFNet(unittest.TestCase):
    def test_mfnet(self):
        mfnet = MFNetNoSigmoid(in_channels = 1, out_channels = 16)
        x = torch.randn(2, 1, 320, 999)
        out = mfnet(x)
        padded_x = nn.functional.pad(x, (0, 16 - x.shape[-1] % 16))
        assert padded_x.size() == out.size()
        pass

if __name__ == "__main__":
    unittest.main()