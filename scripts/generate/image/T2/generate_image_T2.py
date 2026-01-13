import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.utils as vutils
import dnnlib
from torch_utils import distributed as dist
from torch.utils.data import Dataset
from training.dataset_mask import make_rgb
from pathlib import Path
from training.networks_image import register_attention
from torch_utils.io import load_grayscale,load_multiband,save_image
#----------------------------------------------------------------------------
# Configuration presets.

model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions'

config_presets = {
    'edm2-img512-xs-fid':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.135.pkl'),  # fid = 3.53
    'edm2-img512-s-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.130.pkl'),   # fid = 2.56
    'edm2-img512-m-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.100.pkl'),   # fid = 2.25
    'edm2-img512-l-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.085.pkl'),   # fid = 2.06
    'edm2-img512-xl-fid':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.085.pkl'),  # fid = 1.96
    'edm2-img512-xxl-fid':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.070.pkl'), # fid = 1.91
    'edm2-img64-s-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.075.pkl'),    # fid = 1.58
    'edm2-img64-m-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img64-m-2147483-0.060.pkl'),    # fid = 1.43
    'edm2-img64-l-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img64-l-1073741-0.040.pkl'),    # fid = 1.33
    'edm2-img64-xl-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img64-xl-0671088-0.040.pkl'),   # fid = 1.33
    'edm2-img512-xs-dino':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.200.pkl'),  # fd_dinov2 = 103.39
    'edm2-img512-s-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.190.pkl'),   # fd_dinov2 = 68.64
    'edm2-img512-m-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.155.pkl'),   # fd_dinov2 = 58.44
    'edm2-img512-l-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.155.pkl'),   # fd_dinov2 = 52.25
    'edm2-img512-xl-dino':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.155.pkl'),  # fd_dinov2 = 45.96
    'edm2-img512-xxl-dino':      dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.150.pkl'), # fd_dinov2 = 42.84
    'edm2-img512-xs-guid-fid':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.045.pkl',   gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.045.pkl', guidance=1.4), # fid = 2.91
    'edm2-img512-s-guid-fid':    dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.025.pkl',    gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.025.pkl', guidance=1.4), # fid = 2.23
    'edm2-img512-m-guid-fid':    dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.030.pkl',    gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance=1.2), # fid = 2.01
    'edm2-img512-l-guid-fid':    dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.015.pkl',    gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.2), # fid = 1.88
    'edm2-img512-xl-guid-fid':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.020.pkl',   gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.020.pkl', guidance=1.2), # fid = 1.85
    'edm2-img512-xxl-guid-fid':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',  gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.2), # fid = 1.81
    'edm2-img512-xs-guid-dino':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.150.pkl',   gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.150.pkl', guidance=1.7), # fd_dinov2 = 79.94
    'edm2-img512-s-guid-dino':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.085.pkl',    gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.085.pkl', guidance=1.9), # fd_dinov2 = 52.32
    'edm2-img512-m-guid-dino':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.015.pkl',    gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=2.0), # fd_dinov2 = 41.98
    'edm2-img512-l-guid-dino':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.035.pkl',    gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.035.pkl', guidance=1.7), # fd_dinov2 = 38.20
    'edm2-img512-xl-guid-dino':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.030.pkl',   gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance=1.7), # fd_dinov2 = 35.67
    'edm2-img512-xxl-guid-dino': dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',  gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.7), # fd_dinov2 = 33.09
}

#----------------------------------------------------------------------------




def dilate_mask(mask, iterations=1):
    kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32, device=mask.device)
    dilated_mask = mask
    for _ in range(iterations):
        dilated_mask = F.conv2d(dilated_mask, kernel, padding=1)
        dilated_mask = torch.clamp(dilated_mask, max=1)
    return dilated_mask
    


class_rgb_oem = {
    "unknown": [0, 0, 0],
    "Bareland": [128, 0, 0],
    "Grass": [0, 255, 36],
    "Pavement": [148, 148, 148],
    "Road": [255, 255, 255],
    "Tree": [34, 97, 38],
    "Water": [0, 69, 255],
    "Cropland": [75, 181, 73],
    "buildings": [222, 31, 7],
}

class_grey_oem = {k: i for i, k in enumerate(class_rgb_oem)}


def rgb_to_class_mask(rgb_image, class_rgb_oem, class_grey_oem):
    classes = list(class_rgb_oem.keys())
    class_rgb = torch.tensor(list(class_rgb_oem.values()), dtype=torch.float32, device=rgb_image.device)  # (C, 3)

    rgb_image = rgb_image.to(torch.float32)  

    distances = torch.cdist(rgb_image.reshape(-1, 3), class_rgb)  # (B*H*W, C)
    nearest_class_indices = torch.argmin(distances, dim=-1)  # (B*H*W)

    class_grey_values = torch.tensor(
        [class_grey_oem[c] for c in classes], device=rgb_image.device
    )  # (C,)
    
    class_mask = class_grey_values.index_select(0, nearest_class_indices).view(rgb_image.shape[:-1])  # (B, H, W)
    
    return class_mask

class LabelDataset(Dataset):
    def __init__(self, root_dir, t1_dir):

        cond_dir = os.path.join(root_dir)
        t1_msk_dir = os.path.join(t1_dir, 'msk')
        t1_img_latent_dir = os.path.join(t1_dir, 'img_latent')
        t1_msk_latent_dir = os.path.join(t1_dir, 'msk_latent')

        self.cond_files = [os.path.join(cond_dir, f) for f in os.listdir(cond_dir) if f.endswith('.tif')]
        base_names = [os.path.splitext(os.path.basename(f))[0] for f in self.cond_files]
        
        self.t1_img_latent_files = [
            os.path.join(t1_img_latent_dir, name + ".npy")
            for name in base_names
        ]
        self.t1_msk_latent_files = [
            os.path.join(t1_msk_latent_dir, name + ".npy")
            for name in base_names
        ]
        self.t1_msk_files = [
            os.path.join(t1_msk_dir, name + ".tif")
            for name in base_names
        ]

    def __len__(self):
        return len(self.cond_files)

    def __getitem__(self, idx):

        cond_file_path = self.cond_files[idx]
        t1_img_latent_file_path = self.t1_img_latent_files[idx]
        t1_msk_latent_file_path = self.t1_msk_latent_files[idx]
        t1_msk_file_path = self.t1_msk_files[idx]

        file_name = Path(cond_file_path).stem

        t1_img_latent_np = np.load(t1_img_latent_file_path).copy()
        t1_img_latent = torch.from_numpy(t1_img_latent_np[0]).float() # 形状 (C, H, W) 或 (H, W)

        t1_msk_latent_np = np.load(t1_msk_latent_file_path).copy()
        t1_msk_latent = torch.from_numpy(t1_msk_latent_np[0]).float() # 形状 (C, H, W) 或 (H, W)

        t1_msk = load_grayscale(t1_msk_file_path)[::8, ::8].copy()
        t1_msk = torch.from_numpy(t1_msk).unsqueeze(0).long()  # 形状 (1, H, W)

        cond = load_grayscale(cond_file_path).copy()
        cond = torch.from_numpy(cond).unsqueeze(0).long()  # 形状 (1, H, W)

        return dict(
            label=cond, 
            t1_img_latent=t1_img_latent, 
            t1_msk_latent=t1_msk_latent, 
            t1_msk=t1_msk, 
            file_name=file_name
        )

    def get_batch(self, idxs):
        num_files = len(self.cond_files)
        # 索引取模防止溢出
        idxs = [idx % num_files for idx in idxs]
        batch = [self[i] for i in idxs]
        
        # 聚合 batch
        labels = torch.stack([b["label"] for b in batch])          # (B, 1, H, W)
        t1_img_latent = torch.stack([b["t1_img_latent"] for b in batch]) # (B, C, H, W)
        t1_mask_latent = torch.stack([b["t1_msk_latent"] for b in batch]) # (B, C, H, W)
        t1_msk = torch.stack([b["t1_msk"] for b in batch])         # (B, 1, H, W)
        names = [b["file_name"] for b in batch]

        return dict(
            label=labels, 
            t1_img_latent=t1_img_latent, 
            t1_msk_latent=t1_mask_latent, 
            t1_msk=t1_msk, 
            name=names
        )
#----------------------------------------------------------------------------

def edm_sampler(
    net, noise, diff_mask=None, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
    latents=None,
):  
    # Guided denoiser.
    def denoise(x, t):
        Dx = net(x, t, labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t, labels).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    diff_mask = dilate_mask(diff_mask, iterations=2)
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
    x_next = latents.to(dtype)*(1-diff_mask) + noise*t_steps[0]
    # Main sampling loop.
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        # Increase noise temporarily.
        x_cur = x_next
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
            # x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur
        # Euler step.
        Dx = denoise(x_hat, t_hat)
        d_cur = (x_hat - Dx) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        # Apply 2nd order correction.
        if i < num_steps - 1:
            Dx = denoise(x_hat, t_hat)
            d_prime = (x_next -Dx) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        x_next = (latents + torch.randn_like(x_next)*t_next)*(1-diff_mask) + x_next*diff_mask
    return x_next
#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.
class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, index=None,**kwargs):
        if index is None:
            index = torch.ones(size[0], dtype=torch.bool)
        selected_generators = [gen for gen, mask in zip(self.generators, index) if mask]
        assert size[0] == len(selected_generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in selected_generators])

    def randn_like(self, input, index=None):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device, index=index)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------

def generate_images(
    net,                                        # Main network. Path, URL, or torch.nn.Module.
    gnet                = None,                 # Reference network for guidance. None = same as main network.
    encoder_img         = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    segdir              = None,                 # Directory containing condition masks.
    pre_latent_dir      = None,                 # Directory for pre-change latents
    seeds               = range(16, 24),        # List of random seeds.
    max_batch_size      = 32,                   # Maximum batch size for the diffusion model.
    encoder_batch_size  = 4,                    # Maximum batch size for the encoder. None = default.
    verbose             = True,                 # Enable status prints?
    device              = torch.device('cuda'), # Which compute device to use.
    sampler_fn          = edm_sampler,          # Which sampler function to use.
    **sampler_kwargs,                           # Additional arguments for the sampler function.
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load main network.
    if isinstance(net, str):
        if verbose:
            dist.print0(f'Loading network from {net} ...')
        with dnnlib.util.open_url(net, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        # net = data['ema']#.state_dict()
        if isinstance(net, str):
            if verbose:
                dist.print0(f'Loading main network from {net} ...')
            with dnnlib.util.open_url(net, verbose=(verbose and dist.get_rank() == 0)) as f:
                data = pickle.load(f)
            net = data['ema'].to(device)
            net.to(torch.float32)  # 确保 EMA 也转换为 float32
        encoder_img = dnnlib.util.construct_class_by_name(class_name='training.encoders_image.StabilityVAEEncoder')
        encoder_msk = dnnlib.util.construct_class_by_name(class_name='training.encoders_mask.StabilityVAEEncoder')
    assert net is not None

    # Load guidance network.
    if isinstance(gnet, str):
        if verbose:
            dist.print0(f'Loading guidance network from {gnet} ...')
        with dnnlib.util.open_url(gnet, verbose=(verbose and dist.get_rank() == 0)) as f:
            gnet = pickle.load(f)['ema'].to(device)
            gnet.to(torch.float32)  # 确保 EMA 也转换为 float32
    if gnet is None:
        gnet = net
    # Initialize encoder.
    assert encoder_img is not None
    register_attention(net)
    # register_attention(gnet)

    if verbose:
        dist.print0(f'Setting up {type(encoder_img).__name__}...')

    if encoder_batch_size is not None and hasattr(encoder_img, 'batch_size'):
        encoder_img.batch_size = encoder_batch_size

    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')

    dataset = LabelDataset(root_dir=segdir,t1_dir=pre_latent_dir)
    class ImageIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            # Loop over batches.
            for batch_idx, indices in enumerate(rank_batches):
                r = dnnlib.EasyDict(images=None, labels=None, noise=None, batch_idx=batch_idx, num_batches=len(rank_batches), indices=indices)
                r.seeds = [seeds[idx] for idx in indices]               
                if len(r.seeds) > 0:
                    # Pick noise and labels.
                    rnd = StackedRandomGenerator(device, r.seeds)
                    batch_seg = dataset.get_batch(r.seeds)
                    r.noise = rnd.randn([len(r.seeds),net.img_channels,net.img_resolution,net.img_resolution], device=device)
                    # Generate images.
                    t2_msk = batch_seg['label'].to(device=device,dtype=torch.float32)
                    r.labels = t2_msk
                    t1_img_latent = batch_seg['t1_img_latent'].to(device=device,dtype=torch.float32)
                    t1_msk_latent = batch_seg['t1_msk_latent'].to(device=device,dtype=torch.float32)
                    t1_msk = batch_seg['t1_msk'].to(device=device,dtype=torch.float32)
                    r.names = batch_seg['name']
                    t1_img_latent = encoder_img.encode_latents(t1_img_latent)
                    t1_msk_latent = encoder_msk.encode_latents(t1_msk_latent)
                    diff_mask = (t1_msk != t2_msk).to(torch.float32)
                    diff_mask_blend = torch.cat([diff_mask,torch.zeros_like(diff_mask)],dim=0)
                    r.noise_blend  = r.noise.repeat(2,1,1,1)
                    r.latents_blend = torch.cat([t1_img_latent,t1_msk_latent],dim=1).repeat(2,1,1,1)
                    r.labels_blend = torch.cat([t2_msk,t1_msk],dim=0)
        
                    latents = dnnlib.util.call_func_by_name(func_name=sampler_fn, net=net, noise=r.noise_blend, diff_mask=diff_mask_blend,
                    labels=r.labels_blend, gnet=gnet, randn_like=rnd.randn_like, **sampler_kwargs,latents=r.latents_blend)
                    latents, _ = torch.chunk(latents, chunks=2, dim=0)
                    latents_img, latents_msk = torch.chunk(latents, chunks=2, dim=1)
                    msk_hr  = encoder_msk.decode(latents_msk)
                    msk_hr_lr = msk_hr[:,:,::8,::8].permute(0,2,3,1)
                    msk_hr_lr = rgb_to_class_mask(msk_hr_lr, class_rgb_oem, class_grey_oem)
    
                    r.msk_hr  = msk_hr.permute(0,2,3,1)
                    r.msk_hr = rgb_to_class_mask(r.msk_hr, class_rgb_oem, class_grey_oem)
                    r.images = encoder_img.decode(latents_img).permute(0, 2, 3, 1)

                    r.images_t1 = encoder_img.decode(t1_img_latent).permute(0, 2, 3, 1)
                    r.msk_hr_t1  = encoder_msk.decode(t1_msk_latent).permute(0, 2, 3, 1)
                    r.labels = r.labels.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)[:,0]
                    r.names = [name for i, name in enumerate(r.names)]
                    if outdir is not None:
                        sub_dirs = {
                            "img": os.path.join(outdir, "img"),
                            "msk": os.path.join(outdir, "msk"),
                            "demo": os.path.join(outdir, "demo")
                        }
                        for path in sub_dirs.values():
                            os.makedirs(path, exist_ok=True)

                        for seed, image, cond, t2_msk, t1_img, t1_msk, name in zip(
                            r.seeds, r.images.cpu().numpy(), r.labels.cpu().numpy(),
                            r.msk_hr.cpu().numpy(), r.images_t1.cpu().numpy(),
                            r.msk_hr_t1.cpu().numpy(), r.names
                        ):
                            img = save_image(image, os.path.join(sub_dirs["img"], f"{name}.tif"))
                            t2_msk_l = save_image(t2_msk.astype(np.uint8), os.path.join(sub_dirs["msk"], f"{name}.tif"), mode='L')
                            t1_img = save_image(t1_img)  
                            t1_msk = save_image(t1_msk)  
                            cond = save_image(make_rgb(cond))
                            if batch_idx % 1 == 0:

                                t2_msk_rgb = save_image(make_rgb(t2_msk))
                                width = max(img.width, t1_img.width) + max(t2_msk_rgb.width, t1_msk.width) + cond.width
                                height = max(img.height, t2_msk_rgb.height) + max(t1_img.height, t1_msk.height)

                                combined = Image.new('RGB', (width, height))
                                combined.paste(img, (0, 0))  
                                combined.paste(t2_msk_rgb, (img.width, 0)) 
                                combined.paste(cond, (img.width + max(t2_msk_rgb.width, t1_msk.width), 0))  
                                combined.paste(t1_img, (0, img.height))  
                                combined.paste(t1_msk, (img.width, img.height)) 

                                combined.save(os.path.join(sub_dirs["demo"], f"{name}.png"))

                torch.distributed.barrier() # keep the ranks in sync
                yield r

    return ImageIterable()

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--preset',                   help='Configuration preset', metavar='STR',                             type=str, default=None)
@click.option('--net',                      help='Network pickle filename', metavar='PATH|URL',                     type=str, default=None)
@click.option('--gnet',                     help='Reference network for guidance', metavar='PATH|URL',              type=str, default=None)
@click.option('--segdir',                   help='Directory containing condition masks', metavar='DIR',             type=str, required=True)
@click.option('--pre_latent_dir',           help='Directory for pre-change latents', metavar='DIR',                 type=str, required=True)
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, required=True)
@click.option('--seeds',                    help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='16-19', show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=32, show_default=True)

@click.option('--steps', 'num_steps',       help='Number of sampling steps', metavar='INT',                         type=click.IntRange(min=1), default=100, show_default=True)
@click.option('--sigma_min',                help='Lowest noise level', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma_max',                help='Highest noise level', metavar='FLOAT',                            type=click.FloatRange(min=0, min_open=True), default=80, show_default=True)
@click.option('--rho',                      help='Time step exponent', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--guidance',                 help='Guidance strength  [default: 1; no guidance]', metavar='FLOAT',   type=float, default=2.05)
@click.option('--S_churn', 'S_churn',       help='Stochasticity strength', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',           help='Stoch. min noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',           help='Stoch. max noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',       help='Stoch. noise inflation', metavar='FLOAT',                         type=float, default=1, show_default=True)

def cmdline(preset, **opts):
    """Generate random images using the given model.

    Examples:

    \b
    # Generate a couple of images and save them as out/*.png
    python generate_images.py --preset=edm2-img512-s-guid-dino --outdir=out

    \b
    # Generate 50000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 generate_images.py \\
        --preset=edm2-img64-s-fid --outdir=out --subdirs --seeds=0-49999
    """
    opts = dnnlib.EasyDict(opts)

    # Apply preset.
    if preset is not None:
        if preset not in config_presets:
            raise click.ClickException(f'Invalid configuration preset "{preset}"')
        for key, value in config_presets[preset].items():
            if opts[key] is None:
                opts[key] = value

    # Validate options.
    if opts.net is None:
        raise click.ClickException('Please specify either --preset or --net')
    if opts.guidance is None or opts.guidance == 1:
        opts.guidance = 1
        opts.gnet = None
    # elif opts.gnet is None:
    #     raise click.ClickException('Please specify --gnet when using guidance')

    # Generate.
    dist.init()
    image_iter = generate_images(**opts)
    for _r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass

#----------------------------------------------------------------------------
if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------