import os
import re
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image 
import rasterio
import dnnlib
from torch_utils.mask_vis import rgb_to_class_mask
from torch_utils import distributed as dist
from torch_utils.io import load_grayscale,save_image
from torch.utils.data import Dataset

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


class LabelDataset(Dataset):
    def __init__(self, root_dir):
        """
        :param root_dir: Path to the dataset root
        """
        self.files = sorted(
            [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.tif')]
        )
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        :param idx: Index of the data sample
        """
        file_path = self.files[idx]
        msk = load_grayscale(file_path)
        if msk.shape[:2] == (512, 512):
            msk = msk[::8, ::8]
        msk = torch.from_numpy(msk).unsqueeze(0).long()  # 形状 (H, W)
        # one-hot 编码
        return dict(label=msk)
    def get_batch(self, idxs):
        batch = [self[i] for i in idxs]
        labels = torch.stack([b["label"] for b in batch])  # (B, H, W)
        return dict(label=labels)
#----------------------------------------------------------------------------
# EDM sampler from the paper
# "Elucidating the Design Space of Diffusion-Based Generative Models",
# extended to support classifier-free guidance.

def edm_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
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
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        d_cur = (x_hat - denoise(x_hat, t_hat)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next, t_next)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

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
            net.to(torch.float32)  

        encoder_img = dnnlib.util.construct_class_by_name(class_name='training.encoders_image.StabilityVAEEncoder')
        encoder_msk = dnnlib.util.construct_class_by_name(class_name='training.encoders_mask.StabilityVAEEncoder')

    assert net is not None

    # Load guidance network.
    if isinstance(gnet, str):
        if verbose:
            dist.print0(f'Loading guidance network from {gnet} ...')
        with dnnlib.util.open_url(gnet, verbose=(verbose and dist.get_rank() == 0)) as f:
            gnet = pickle.load(f)['ema'].to(device)
            gnet.to(torch.float32)  
    if gnet is None:
        gnet = net
    # Initialize encoder.
    assert encoder_img is not None

    if verbose:
        # dist.print0(f'Setting up {type(encoder_img).__name__}...')
        dist.print0(f'Setting up {type(encoder_img).__name__}...')

    if encoder_batch_size is not None and hasattr(encoder_img, 'batch_size'):
        encoder_img.batch_size = encoder_batch_size

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')

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
                    batch_seg = dataset.get_batch(r.seeds)
                    r.noise = rnd.randn([len(r.seeds),net.img_channels,net.img_resolution,net.img_resolution], device=device)
                    # Generate images.
                    r.labels = batch_seg['label'].to(device=device,dtype=torch.float32)
                    latents = dnnlib.util.call_func_by_name(func_name=sampler_fn, net=net, noise=r.noise,
                        labels=r.labels, gnet=gnet, randn_like=rnd.randn_like, **sampler_kwargs)
                    latents_img, latents_msk = torch.chunk(latents, chunks=2, dim=1)
                    r.images = encoder_img.decode(latents_img).permute(0, 2, 3, 1)
                    r.masks  = encoder_msk.decode(latents_msk).permute(0, 2, 3, 1)
                    r.masks = rgb_to_class_mask(r.masks)
                    if outdir is not None:
                        for seed, image,mask in zip(r.seeds, r.images.cpu().numpy(),r.masks.cpu().numpy()):
                            image_dir = os.path.join(outdir,'img')
                            mask_dir = os.path.join(outdir, 'msk')
                            os.makedirs(image_dir, exist_ok=True)
                            os.makedirs(mask_dir, exist_ok=True)
                            save_image(image, os.path.join(image_dir, f'{seed:06d}.tif'))
                            save_image(mask.astype(np.uint8), os.path.join(mask_dir, f'{seed:06d}.tif'), mode='L')
                # Yield results.
                torch.distributed.barrier() # keep the ranks in sync
                yield r

    return ImageIterable()


#----------------------------------------------------------------------------
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
@click.option('--segdir',                   help='Where to save the seg images', metavar='DIR',                     type=str, required=True)
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, required=True)
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                        is_flag=True)
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