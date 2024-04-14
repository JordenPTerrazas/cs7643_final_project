import unittest
import torch
import torch.nn as nn
from GLFB import GLFB
from UpSample import UpSample
from DownSample import DownSample
from Projection import Projection

# One thing not specified in the paper is the scenario where we end up with rounding
# error when we downsample and upsample. I'm going to assume that we should pad the
# result at the layer it occurs, but not sure on best practice.
class MFNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 16):
        super(MFNet, self).__init__()
        
        # Projection Layers (in_channels -> out_channels)
        self.projection_1 = Projection(
            in_channels = in_channels, 
            out_channels = out_channels
        )

        self.projection_2 = Projection(
            in_channels = out_channels,
            out_channels = in_channels
        )

        # GLFB Layers (out_channels -> out_channels)
        self.glfb_1 = GLFB(
            in_channels = out_channels,
            frame_size = 320,
            n_frames = 1008
        )

        self.glfb_2 = GLFB(
            in_channels = out_channels * 2,
            frame_size = 160,
            n_frames = 504
        )

        self.glfb_3_1 = GLFB(
            in_channels = out_channels * 4,
            frame_size = 80,
            n_frames = 252
        )

        self.glfb_3_2 = GLFB(
            in_channels = out_channels * 4,
            frame_size = 80,
            n_frames = 252
        )

        self.glfb_3_3 = GLFB(
            in_channels = out_channels * 4,
            frame_size = 80,
            n_frames = 252
        )

        self.glfb_3_4 = GLFB(
            in_channels = out_channels * 4,
            frame_size = 80,
            n_frames = 252
        )

        self.glfb_3_5 = GLFB(
            in_channels = out_channels * 4,
            frame_size = 80,
            n_frames = 252
        )

        self.glfb_3_6 = GLFB(
            in_channels = out_channels * 4,
            frame_size = 80,
            n_frames = 252
        )

        self.glfb_3_7 = GLFB(
            in_channels = out_channels * 4,
            frame_size = 80,
            n_frames = 252
        )

        self.glfb_3_8 = GLFB(
            in_channels = out_channels * 4,
            frame_size = 80,
            n_frames = 252
        )

        self.glfb_4_1 = GLFB(
            in_channels = out_channels * 8,
            frame_size = 40,
            n_frames = 126
        )

        self.glfb_4_2 = GLFB(
            in_channels = out_channels * 8,
            frame_size = 40,
            n_frames = 126
        )

        self.glfb_4_3 = GLFB(
            in_channels = out_channels * 8,
            frame_size = 40,
            n_frames = 126
        )

        self.glfb_4_4 = GLFB(
            in_channels = out_channels * 8,
            frame_size = 40,
            n_frames = 126
        )

        self.glfb_5_1 = GLFB(
            in_channels = out_channels * 16,
            frame_size = 20,
            n_frames = 63
        )

        self.glfb_5_2 = GLFB(
            in_channels = out_channels * 16,
            frame_size = 20,
            n_frames = 63
        )

        self.glfb_5_3 = GLFB(
            in_channels = out_channels * 16,
            frame_size = 20,
            n_frames = 63
        )

        self.glfb_5_4 = GLFB(
            in_channels = out_channels * 16,
            frame_size = 20,
            n_frames = 63
        )

        self.glfb_5_5 = GLFB(
            in_channels = out_channels * 16,
            frame_size = 20,
            n_frames = 63
        )

        self.glfb_5_6 = GLFB(
            in_channels = out_channels * 16,
            frame_size = 20,
            n_frames = 63
        )

        self.glfb_6 = GLFB(
            in_channels = out_channels * 8,
            frame_size = 40,
            n_frames = 126
        )

        self.glfb_7 = GLFB(
            in_channels = out_channels * 4,
            frame_size = 80,
            n_frames = 252
        )

        self.glfb_8 = GLFB(
            in_channels = out_channels * 2,
            frame_size = 160,
            n_frames = 504
        )

        self.glfb_9 = GLFB(
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
        
        x = self.downsample_1(x1)
        x2 = self.glfb_2(x)

        x = self.downsample_2(x2)
        x = self.glfb_3_1(x)
        x = self.glfb_3_2(x)
        x = self.glfb_3_3(x)
        x = self.glfb_3_4(x)
        x = self.glfb_3_5(x)
        x = self.glfb_3_6(x)
        x = self.glfb_3_7(x)
        x3 = self.glfb_3_8(x)

        x = self.downsample_3(x3)
        x = self.glfb_4_1(x)
        x = self.glfb_4_2(x)
        x = self.glfb_4_3(x)
        x4 = self.glfb_4_4(x)

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
        
        # Decoder part
        x = self.glfb_6(x)
        x = self.upsample_2(x)
        x = torch.add(x, x3)

        x = self.glfb_7(x)
        x = self.upsample_3(x)
        x = torch.add(x, x2)

        x = self.glfb_8(x)
        x = self.upsample_4(x)
        x = torch.add(x, x1)

        x = self.glfb_9(x)
        x = self.projection_2(x)
        return x

class TestMFNet(unittest.TestCase):
    def test_mfnet(self):
        mfnet = MFNet(in_channels = 1, out_channels = 16)
        x = torch.randn(1, 320, 999)
        out = mfnet(x)
        padded_x = nn.functional.pad(x, (0, 16 - x.shape[-1] % 16))
        assert padded_x.size() == out.size()
        pass

if __name__ == "__main__":
    unittest.main()