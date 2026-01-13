# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Converting between pixel and latent representations of image data."""

import os
import warnings
import numpy as np
import torch
from torch_utils import persistence
from torch_utils import misc

warnings.filterwarnings('ignore', 'torch.utils._pytree._register_pytree_node is deprecated.')
warnings.filterwarnings('ignore', '`resume_download` is deprecated')

#----------------------------------------------------------------------------
# Abstract base class for encoders/decoders that convert back and forth
# between pixel and latent representations of image data.
#
# Logically, "raw pixels" are first encoded into "raw latents" that are
# then further encoded into "final latents". Decoding, on the other hand,
# goes directly from the final latents to raw pixels. The final latents are
# used as inputs and outputs of the model, whereas the raw latents are
# stored in the dataset. This separation provides added flexibility in terms
# of performing just-in-time adjustments, such as data whitening, without
# having to construct a new dataset.
#
# All image data is represented as PyTorch tensors in NCHW order.
# Raw pixels are represented as 3-channel uint8.

@persistence.persistent_class
class Encoder:
    def __init__(self):
        pass

    def init(self, device): # force lazy init to happen now
        pass

    def __getstate__(self):
        return self.__dict__

    def encode(self, x): # raw pixels => final latents
        return self.encode_latents(self.encode_pixels(x))

    def encode_pixels(self, x): # raw pixels => raw latents
        raise NotImplementedError # to be overridden by subclass

    def encode_latents(self, x): # raw latents => final latents
        raise NotImplementedError # to be overridden by subclass

    def decode(self, x): # final latents => raw pixels
        raise NotImplementedError # to be overridden by subclass

#----------------------------------------------------------------------------
# Standard RGB encoder that scales the pixel data into [-1, +1].

@persistence.persistent_class
class StandardRGBEncoder(Encoder):
    def __init__(self):
        super().__init__()

    def encode_pixels(self, x): # raw pixels => raw latents
        return x

    def encode_latents(self, x): # raw latents => final latents
        return x.to(torch.float32) / 127.5 - 1

    def decode(self, x): # final latents => raw pixels
        return (x.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)

@persistence.persistent_class
class StandardRGBEncoder(Encoder):
    def __init__(self):
        super().__init__()

    def encode_pixels(self, x): # raw pixels => raw latents
        return x

    def encode_latents(self, x): # raw latents => final latents
        return x.to(torch.float32) / 127.5 - 1

    def decode(self, x): # final latents => raw pixels
        return (x.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)


@persistence.persistent_class
class DiscreteEncoder(Encoder):
    def __init__(self):
        super().__init__()

    def encode_pixels(self, x: torch.Tensor):
        x = x.to(torch.int32)  # 确保输入为整数类型
        shifts = torch.arange(3, -1, -1, device=x.device).view(1, 4, 1, 1)  # 位移值 [3, 2, 1, 0]
        bitmap = (torch.bitwise_right_shift(x.unsqueeze(1), shifts) & 1).float()  # 提取比特，(B, 4, H, W)
        return bitmap*2-1  # 变成 (B, H, W, 4)

    # 解码过程：将4位二进制比特转换回整数
    def decode_pixels(self, bitmap: torch.Tensor):
        bitmap = (bitmap+1)/2
        powers_of_two = torch.tensor([8, 4, 2, 1], device=bitmap.device, dtype=bitmap.dtype)  # 对应2^3, 2^2, 2^1, 2^0
        return torch.sum(bitmap * powers_of_two.view(1, 4, 1, 1), dim=1,keepdim=True).long()  # 按照最后一个维度加权求和
    
    def decode_noise(self, bitmap: torch.Tensor):
        bitmap = (bitmap+1)/2
        powers_of_two = torch.tensor([8, 4, 2, 1], device=bitmap.device, dtype=bitmap.dtype)  # 对应2^3, 2^2, 2^1, 2^0
        return torch.sum(bitmap * powers_of_two.view(1, 4, 1, 1), dim=1,keepdim=True).long()  # 按照最后一个维度加权求和



# 编码过程：将整数转化为4位二进制



#----------------------------------------------------------------------------
# Pre-trained VAE encoder from Stability AI.
#OpenMap
# raw_mean    = [0.25717577,0.2907192, 0.9108262, 0.64083916],    # Assumed mean of the raw latents.
# raw_std     = [3.8868818, 3.8952892, 3.4912703, 3.3273928],     # Assumed standard deviation of the raw latents.
#OpenMap
#MASK




@persistence.persistent_class
class StabilityVAEEncoder(Encoder):
    def __init__(self,
        vae_name    = "./pretrained_vae/sd-vae-ft-mse",  # Name of the VAE to use.
        raw_mean = [ 4.3026466,  3.6255362,  0.5283355, -1.7482258],
        raw_std = [3.5087192, 3.4944136, 3.250516,  2.7732227],
        final_mean  = 0,                            # Desired mean of the final latents.
        final_std   = 0.5,                          # Desired standard deviation of the final latents.
        batch_size  = 8,                            # Batch size to use when running the VAE.
    ):
        super().__init__()
        self.vae_name = vae_name
        self.scale = np.float32(final_std) / np.float32(raw_std)
        self.bias = np.float32(final_mean) - np.float32(raw_mean) * self.scale
        self.batch_size = int(batch_size)
        self._vae = None
    def init(self, device): # force lazy init to happen now
        super().init(device)
        if self._vae is None:
            print('')
            self._vae = load_stability_vae(self.vae_name, device=device)
        else:
            self._vae.to(device)

    def __getstate__(self):
        return dict(super().__getstate__(), _vae=None) # do not pickle the vae

    def _run_vae_encoder(self, x):
        d = self._vae.encode(x)['latent_dist']
        return torch.cat([d.mean, d.std], dim=1)

    def _run_vae_decoder(self, x):
        return self._vae.decode(x)['sample']

    def encode_pixels(self, x): # raw pixels => raw latents
        self.init(x.device)
        x = x.to(torch.float32) / 255
        x = torch.cat([self._run_vae_encoder(batch) for batch in x.split(self.batch_size)])
        return x

    def encode_latents(self, x): # raw latents => final latents
        mean, std = x.to(torch.float32).chunk(2, dim=1)
        x = mean + torch.randn_like(mean) * std
        x = x * misc.const_like(x, self.scale).reshape(1, -1, 1, 1)
        x = x + misc.const_like(x, self.bias).reshape(1, -1, 1, 1)
        return x

    def decode(self, x): # final latents => raw pixels
        self.init(x.device)
        x = x.to(torch.float32)
        x = x - misc.const_like(x, self.bias).reshape(1, -1, 1, 1)
        x = x / misc.const_like(x, self.scale).reshape(1, -1, 1, 1)
        x = torch.cat([self._run_vae_decoder(batch) for batch in x.split(self.batch_size)])
        x = x.clamp(0, 1).mul(255).to(torch.uint8)
        return x


#----------------------------------------------------------------------------

def load_stability_vae(vae_name="./pretrained_vae/sd-vae-ft-mse", device=torch.device('cpu')):
    # vae = StabilityVAEEncoder(vae_name=vae_name)
    import diffusers # pip install diffusers # pyright: ignore [reportMissingImports]
    vae = diffusers.models.AutoencoderKL.from_pretrained(vae_name, local_files_only=True)
    return vae.eval().requires_grad_(False).to(device)

#----------------------------------------------------------------------------
