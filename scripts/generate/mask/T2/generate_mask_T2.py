import os
import re
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import dnnlib
import glob
# import diffusers
from torch_utils import distributed as dist
from torch.utils.data import Dataset
from training.dataset_mask import make_rgb
import cv2
import random
import rasterio
from pathlib import Path
warnings.filterwarnings('ignore', '`resume_download` is deprecated')
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

def remove_random_regions_from_batch(
    masks: np.ndarray,
    target_classes,
    min_ratio: float = 0.1,
    max_ratio: float = 0.5,
) -> np.ndarray:
    """
    Randomly replace connected regions in segmentation masks.

    Args:
        masks: (B, H, W) int numpy array
        target_classes: iterable of class indices
    """
    assert masks.ndim == 3
    B, H, W = masks.shape
    output = masks.copy()

    target_classes = [int(c) for c in target_classes]
    for i in range(B):
        total = H * W
        target_pixels = random.randint(
            int(min_ratio * total),
            int(max_ratio * total),
        )

        regions = []
        for cls in np.unique(output[i]):
            binary = (output[i] == cls).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(binary)
            for lab in range(1, num_labels):
                region = labels == lab
                size = int(region.sum())
                if size > 0:
                    regions.append((size, region))

        random.shuffle(regions)
        removed = 0

        for size, region in regions:
            if removed + size > target_pixels:
                continue
            output[i][region] = random.choice(target_classes)
            removed += size
            if removed >= target_pixels:
                break

    return output

def load_multiband(path: str):
    with rasterio.open(path, "r") as src:
        return (np.moveaxis(src.read(), 0, -1)).astype(np.uint8)

def load_grayscale(path: str):
    with rasterio.open(path, "r") as src:
        return (src.read(1)).astype(np.uint8)

#----------------------------------------------------------------------------
class LabelDataset(Dataset):
    def __init__(self, root_dir):
        """
        :param root_dir: dataset root directory
        """
        self.msk_fps = glob.glob(os.path.join(root_dir, '*.tif'))
        # with open(os.path.join(root_dir, 'xbd_matched.txt'), 'r') as f:
            # self.msk_fps = [line.strip() for line in f if line.strip()]
            # self.msk_fps = [line.strip().replace('t1_images', 't1_masks') for line in f if line.strip()]
        self.ignore_label=0
    def __len__(self):
        return len(self.msk_fps)

    def __getitem__(self, idx):
        """
        Return a sample by index.
        :param idx: data index
        """
        idx = idx % len(self.msk_fps)  # wrap index when out of range
        msk = load_grayscale(self.msk_fps[idx])
        if msk.shape[:2] == (512, 512):
            msk = msk[::8, ::8]
        msk = torch.from_numpy(msk).long()  # shape (H, W)
        file_stem = Path(self.msk_fps[idx]).stem
        # one-hot encoding (if needed)
        return dict(mask=msk,name=file_stem)
    def get_batch(self, idxs):
        idxs = [i % len(self.msk_fps) for i in idxs]  # ensure indices are within range
        batch = [self[i] for i in idxs]
        masks = torch.stack([b["mask"] for b in batch])  # (B, H, W)
        names = [b["name"] for b in batch]  # (B, H, W)
        """Return a batch for given list of indices"""
        return dict(masks=masks,names=names)

class LBSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x + (torch.sign(x) - x).detach()

    @staticmethod
    def backward(ctx, grad):
        return grad.clamp_(-1, 1)

def softsign(x):
    return x / (1 + torch.abs(x))

#----------------------------------------------------------------------------
# EDM sampler from the paper
# "Elucidating the Design Space of Diffusion-Based Generative Models",
# extended to support classifier-free guidance.

def edm_sampler(
    net, noise, mask_t1=None ,labels=None, gnet=None,encoder_img=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=1, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like, num_classes=None,
):

    def denoise(x, t):
        Dx = net(x, t, labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t, labels).to(dtype)
        return ref_Dx.lerp(Dx, guidance)
    
    labels_change = (labels[0] > 0).nonzero(as_tuple=True)[0].cpu().tolist()

    mask_t1_np = mask_t1.cpu().numpy()

    mask_t1_props = torch.stack([torch.bincount(m.flatten().long(), minlength=num_classes).float() / m.numel() for m in mask_t1])
    mask_t2_np = remove_random_regions_from_batch(mask_t1_np, labels_change, min_ratio=0.1, max_ratio=0.5)
    changed_map = np.where(mask_t1_np != mask_t2_np, mask_t1_np, 0)
    changed_map = torch.from_numpy(changed_map).to(device=noise.device, dtype=torch.long)
    changed_binary = (changed_map >= 1).int().unsqueeze(1)
    sign = LBSign.apply

    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
    powers_of_two = torch.tensor([8, 4, 2, 1], device=noise.device, dtype=noise.dtype)  # corresponds to 2^3, 2^2, 2^1, 2^0
    mask_t2 = torch.as_tensor(mask_t2_np, device=noise.device, dtype=torch.int32)
    latent_t2 = encoder_img.encode_pixels(mask_t2)
    x_next = noise.to(dtype) * t_steps[0] + latent_t2
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next.detach()
        if  20<=i<=(num_steps-(num_steps-25)):
            lr=0.1
            def mask_gradients(grad):
                return grad * changed_binary
            x_cur.requires_grad_(True)
            x_cur.register_hook(mask_gradients)
            torch.set_grad_enabled(True)
            optimizer= torch.optim.Adam([x_cur], lr=lr, weight_decay=0.0)
            for opt_stp in range(1):
                latent = denoise(x_cur, t_cur)   
                latent = torch.tanh(latent)
                latent = sign(latent)
                latent = (latent+1)/2
                ################### decode ###################
                mask = torch.sum(latent * powers_of_two.view(1, 4, 1, 1), dim=1,keepdim=True)
                ################### decode ##################
                mask_original = mask.clone()
                mask_original[mask_original >= num_classes] = 0 # other class
                mask_original = (mask_original*changed_binary)

                mask_onehot = F.one_hot(mask_original.long(), num_classes=num_classes).squeeze(1).permute(0, 3, 1, 2).float()
                mask_onehot[:, 0] = mask_onehot[:, 0]*changed_binary.squeeze(1)
                mask = mask/(mask.detach()+1e-9)*mask_onehot
                mask_ratio_t2 = mask.sum(dim=[2,3])[:,:]/(mask_onehot[:,:].sum(dim=[1,2,3]).unsqueeze(1)+1e-6) 
                class_mask = torch.zeros_like(labels)
                for tc in labels_change:
                    class_mask[:, tc] = 1
                class_mask[:, 0] = 1
                loss = torch.mean(((mask_ratio_t2 - 0.5 * mask_t1_props.detach()) * class_mask).pow(2))
                optimizer.zero_grad(set_to_none=True)
                loss.backward(retain_graph=True)
                optimizer.step()
        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur   
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur
        # Euler step.
        #########
        #repaint
        ########
        if i <= 20:

            noise = torch.randn_like(x_next, dtype=x_next.dtype, device=x_next.device) * t_next
            repaint = latent_t2 + noise
            x_next = repaint
        else:
            d_cur = (x_hat - denoise(x_hat, t_hat)) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            noise = torch.randn_like(x_next, dtype=x_next.dtype, device=x_next.device) * t_next
            repaint = latent_t2 + noise
            mask = changed_binary.to(torch.bool)
            x_next = torch.where(mask, x_next, repaint)

            if i < num_steps - 1:
                d_prime = (x_next - denoise(x_next, t_next)) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                noise = torch.randn_like(x_next, dtype=x_next.dtype, device=x_next.device) * t_next
                repaint = latent_t2 + noise
                mask = changed_binary.to(torch.bool)
                x_next = torch.where(mask, x_next, repaint)

    return x_next,changed_binary

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------

def generate_images(
    net,                                        # Main network. Path, URL, or torch.nn.Module.
    gnet                = None,                 # Reference network for guidance. None = same as main network.
    encoder_img         = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    segdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),        # List of random seeds.
    class_idx           = None,                 # Class label. None = select randomly.
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
        net = data['ema']
        net.to(device)
        if encoder_img is None:
            encoder_img = data.get('encoder_img', None)
            if encoder_img is None:
                encoder_img = dnnlib.util.construct_class_by_name(class_name='training.encoders.DiscreteEncoder')

    assert net is not None

    # Load guidance network.
    if isinstance(gnet, str):
        if verbose:
            dist.print0(f'Loading guidance network from {gnet} ...')
        with dnnlib.util.open_url(gnet, verbose=(verbose and dist.get_rank() == 0)) as f:
            gnet = pickle.load(f)['ema'].to(device)
    if gnet is None:
        gnet = net
    # Initialize encoder.
    assert encoder_img is not None
    # assert encoder_msk is not None

    if verbose:
        # dist.print0(f'Setting up {type(encoder_img).__name__}...')
        dist.print0(f'Setting up {type(encoder_img).__name__}...')

    if encoder_batch_size is not None and hasattr(encoder_img, 'batch_size'):
        encoder_img.batch_size = encoder_batch_size

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    def load_grayscale(path: str):
        with rasterio.open(path, "r") as src:
            return src.read(1).astype(np.uint8)

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')

    # Use DataLoader to load data
    dataset = LabelDataset(root_dir=segdir)
    # Return an iterable over the batches.
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
                    r.noise = rnd.randn([len(r.seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
                    batch_seg = dataset.get_batch(r.seeds)
                    r.mask_t1 = batch_seg['masks'].to(device=device,dtype=torch.float32)
                    r.names = batch_seg['names']
                    # Generate images.
                    num_classes = getattr(net, 'label_dim', 9)
                    r.labels = torch.zeros((len(r.seeds), num_classes), dtype=torch.float32).to(device=device)
                    if net.label_dim > 0:
                        r.labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[len(r.seeds)], device=device)]
                        if class_idx is not None:
                            cls_list = class_idx if isinstance(class_idx, (list, tuple)) else [int(class_idx)]
                            for c in cls_list:
                                ci = int(c)
                                if not (0 <= ci < net.label_dim):
                                    raise click.ClickException(f'--class index out of range: {c}')
                            r.labels[:, :] = 0
                            for c in cls_list:
                                r.labels[:, int(c)] = 1
 
                    latents,changemap = dnnlib.util.call_func_by_name(func_name=sampler_fn, net=net, noise=r.noise, 
                        mask_t1=r.mask_t1, labels=r.labels ,gnet=gnet, randn_like=rnd.randn_like,encoder_img=encoder_img, num_classes=num_classes, **sampler_kwargs)
                    latents = torch.sign(latents)
                    r.mask_t2 = encoder_img.decode_pixels(latents).squeeze(1)
                    # Save images.
                    if outdir is not None:
                        for seed, mask_t2, mask_t1,name in zip(r.seeds, r.mask_t2.cpu().numpy(), r.mask_t1.cpu().numpy(), r.names):

                            # ensure output subdirectories exist
                            os.makedirs(os.path.join(outdir, 'msk'), exist_ok=True)
                            os.makedirs(os.path.join(outdir, 'demo'), exist_ok=True)
                            mask_t2_L = Image.fromarray(mask_t2.astype(np.uint8), 'L')
                            
                            mask_t2_dir = os.path.join(outdir, 'msk')
                            demo_dir = os.path.join(outdir, 'demo')

                            mask_t2_L.save(os.path.join(mask_t2_dir, f'{name}.tif'))

                            mask_t1 = make_rgb(mask_t1)
                            mask_t2 = make_rgb(mask_t2)

                            mask_t2 = Image.fromarray(mask_t2, 'RGB')
                            mask_t1 = Image.fromarray(mask_t1, 'RGB')
                            # create combined demo image
                            width_t2, height_t2 = mask_t2.size
                            width_t1, height_t1 = mask_t1.size

                            combined_width = width_t2 + width_t1
                            combined_height = max(height_t2, height_t1)
                            combined_image = Image.new('RGB', (combined_width, combined_height))
                            combined_image.paste(mask_t2, (0, 0))
                            combined_image.paste(mask_t1, (width_t2, 0))
                            combined_image.save(os.path.join(demo_dir, f'{name}.png'))
                # Yield results.
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
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, required=True)
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                        is_flag=True)
@click.option('--seeds',                    help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='16-19', show_default=True)
@click.option('--class', 'class_idx',       help='Class label or comma-separated class list (e.g. 1 or 1,2,3)',     metavar='LIST', type=str, default='8')
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--segdir',                   help='Where to save the seg images', metavar='DIR',                     type=str, required=True)

@click.option('--steps', 'num_steps',       help='Number of sampling steps', metavar='INT',                         type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--sigma_min',                help='Lowest noise level', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma_max',                help='Highest noise level', metavar='FLOAT',                            type=click.FloatRange(min=0, min_open=True), default=80, show_default=True)
@click.option('--rho',                      help='Time step exponent', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--guidance',                 help='Guidance strength  [default: 1; no guidance]', metavar='FLOAT',   type=float, default=2.45)
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
    # Normalize `class_idx`: accept single int string like '3' or comma-separated '1,2,3'
    if isinstance(opts.class_idx, str):
        try:
            if ',' in opts.class_idx:
                opts.class_idx = [int(x) for x in opts.class_idx.split(',') if x.strip() != '']
            else:
                opts.class_idx = int(opts.class_idx)
        except Exception:
            raise click.ClickException('--class must be an int or comma-separated list of ints')

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