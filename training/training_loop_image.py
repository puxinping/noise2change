# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import re
import time
import copy
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import persistence
from torch_utils import misc


def _should_use_cpu_gloo_grad_sync(device):
    return dist.get_world_size() > 1 and device.type == 'cuda'


@torch.no_grad()
def _sync_module_grads_via_cpu(module, sync_group):
    params = [param for param in module.parameters() if param.grad is not None]
    if len(params) == 0:
        return

    flat_grad = torch.cat([param.grad.detach().to(device='cpu', dtype=torch.float32).reshape(-1) for param in params])
    torch.distributed.all_reduce(flat_grad, op=torch.distributed.ReduceOp.SUM, group=sync_group)
    flat_grad.div_(dist.get_world_size())

    offset = 0
    for param in params:
        numel = param.grad.numel()
        param.grad.copy_(flat_grad[offset : offset + numel].view_as(param.grad).to(device=param.grad.device, dtype=param.grad.dtype))
        offset += numel


@torch.no_grad()
def _check_module_consistency_via_cpu(module, sync_group, ignore_regex=None):
    for name, tensor in misc.named_params_and_buffers(module):
        fullname = type(module).__name__ + '.' + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        local = tensor.detach()
        if local.is_floating_point():
            local = torch.nan_to_num(local)
        local = local.cpu()
        other = local.clone()
        torch.distributed.broadcast(tensor=other, src=0, group=sync_group)
        assert torch.equal(local, other), fullname


def _collect_resolution_tokens(module_dict):
    tokens = []
    for name in module_dict.keys():
        match = re.match(r'(\d+)x\1_', name)
        if match is None:
            continue
        token = match.group(0)[:-1]
        if not tokens or tokens[-1] != token:
            tokens.append(token)
    return tokens


def _bridge_input_conv_weight(src_tensor, dst_tensor):
    bridged = torch.zeros(dst_tensor.shape, dtype=src_tensor.dtype, device=src_tensor.device)
    bridged[:, :4] = src_tensor[:, :4]
    bridged[:, -1:] = src_tensor[:, -1:]
    return bridged


def _bridge_output_conv_weight(src_tensor, dst_tensor):
    bridged = torch.zeros(dst_tensor.shape, dtype=src_tensor.dtype, device=src_tensor.device)
    bridged[:4] = src_tensor
    return bridged


def _build_image_warmstart_state(pretrained_net, target_net):
    src_state_dict = pretrained_net.state_dict()
    dst_state_dict = target_net.state_dict()
    src_tokens = _collect_resolution_tokens(pretrained_net.unet.enc)
    dst_tokens = _collect_resolution_tokens(target_net.unet.enc)
    resolution_map = {src: dst for src, dst in zip(src_tokens, dst_tokens)}

    loadable_state = {}
    direct_keys = []
    remapped_keys = []
    bridged_keys = []
    expanded_emb_linear_keys = []
    skipped_label_keys = []
    skipped_missing = []
    skipped_shape = []

    for src_key, src_tensor in src_state_dict.items():
        dst_key = src_key
        for src_token, dst_token in resolution_map.items():
            dst_key = dst_key.replace(src_token, dst_token)

        tensor_to_load = src_tensor
        if src_key == 'unet.emb_label.weight':
            skipped_label_keys.append(src_key)
            continue
        if dst_key == 'unet.enc.128x128_conv.weight':
            tensor_to_load = _bridge_input_conv_weight(src_tensor=src_tensor, dst_tensor=dst_state_dict[dst_key])
            bridged_keys.append((src_key, dst_key, 'input-conv'))
        elif dst_key == 'unet.out_conv.weight':
            tensor_to_load = _bridge_output_conv_weight(src_tensor=src_tensor, dst_tensor=dst_state_dict[dst_key])
            bridged_keys.append((src_key, dst_key, 'output-conv'))
        elif src_key.endswith('emb_linear.weight') and src_tensor.ndim == 2:
            tensor_to_load = src_tensor[:, :, None, None]
            expanded_emb_linear_keys.append((src_key, dst_key))

        if dst_key not in dst_state_dict:
            skipped_missing.append((src_key, dst_key, tuple(src_tensor.shape)))
            continue
        if tuple(tensor_to_load.shape) != tuple(dst_state_dict[dst_key].shape):
            skipped_shape.append((src_key, dst_key, tuple(tensor_to_load.shape), tuple(dst_state_dict[dst_key].shape)))
            continue

        loadable_state[dst_key] = tensor_to_load
        if dst_key == src_key and torch.equal(tensor_to_load.detach().cpu(), src_tensor.detach().cpu()):
            direct_keys.append(src_key)
        elif any(dst_key == bridged_key for _, bridged_key, _ in bridged_keys):
            pass
        else:
            remapped_keys.append((src_key, dst_key))

    return dnnlib.EasyDict(
        loadable_state=loadable_state,
        resolution_map=resolution_map,
        direct_keys=direct_keys,
        remapped_keys=remapped_keys,
        bridged_keys=bridged_keys,
        expanded_emb_linear_keys=expanded_emb_linear_keys,
        skipped_label_keys=skipped_label_keys,
        skipped_missing=skipped_missing,
        skipped_shape=skipped_shape,
    )


@torch.no_grad()
def _zero_module_parameters(module):
    if module is None:
        return
    for parameter in module.parameters():
        parameter.zero_()


@torch.no_grad()
def _load_image_warmstart(net, ema, pretrained_pkl):
    dist.print0(f'Warm-starting image model from {pretrained_pkl} ...')
    with open(pretrained_pkl, 'rb') as f:
        pretrained_net = pickle.load(f)['ema'].cpu()

    warmstart = _build_image_warmstart_state(pretrained_net=pretrained_net, target_net=net)
    load_result = net.load_state_dict(warmstart.loadable_state, strict=False)
    _zero_module_parameters(getattr(net.unet, 'emb_label', None))

    if ema is not None:
        ema.reset()

    resolution_map_str = ', '.join(f'{src}->{dst}' for src, dst in warmstart.resolution_map.items())
    dist.print0(f'Warm start resolution map: {resolution_map_str}')
    dist.print0(
        'Warm start summary: '
        f'loaded={len(warmstart.loadable_state)} '
        f'(direct={len(warmstart.direct_keys)}, remapped={len(warmstart.remapped_keys)}, '
        f'bridged={len(warmstart.bridged_keys)}, emb_linear_expanded={len(warmstart.expanded_emb_linear_keys)}), '
        f'skipped_label={len(warmstart.skipped_label_keys)}, '
        f'skipped_missing={len(warmstart.skipped_missing)}, '
        f'skipped_shape={len(warmstart.skipped_shape)}'
    )
    if load_result.unexpected_keys:
        dist.print0(f'Warm start unexpected keys: {load_result.unexpected_keys}')
    if load_result.missing_keys:
        dist.print0(f'Warm start missing keys: {load_result.missing_keys}')

#----------------------------------------------------------------------------
# Uncertainty-based loss function (Equations 14,15,16,21) proposed in the
# paper "Analyzing and Improving the Training Dynamics of Diffusion Models".

@persistence.persistent_class
class EDM2Loss:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(images) * sigma
        denoised, logvar = net(images + noise, sigma, labels, return_logvar=True)
        loss = (weight / logvar.exp()) * ((denoised - images) ** 2) + logvar
        return loss

#----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".

def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr

#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
# Main training loop.

def training_loop(
    dataset_kwargs      = dict(class_name='training.dataset.CustomDataset', path=None),
    encoder_img_kwargs  = dict(class_name='training.encoders.StabilityVAEEncoder'),
    encoder_msk_kwargs  = dict(class_name='training.encoders_msk_rgb.StabilityVAEEncoder'),
    data_loader_kwargs  = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=2, prefetch_factor=2),
    network_kwargs      = dict(class_name='training.networks_edm2.Precond'),
    loss_kwargs         = dict(class_name='training.training_loop.EDM2Loss'),
    optimizer_kwargs    = dict(class_name='torch.optim.Adam', betas=(0.9, 0.99)),
    lr_kwargs           = dict(func_name='training.training_loop.learning_rate_schedule'),
    ema_kwargs          = dict(class_name='training.phema.PowerFunctionEMA'),

    run_dir             = '.',      # Output directory.
    seed                = 0,        # Global random seed.
    batch_size          = 2048,     # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU. None = no limit.
    total_nimg          = 8<<30,    # Train for a total of N training images.
    slice_nimg          = None,     # Train for a maximum of N training images in one invocation. None = no limit.
    status_nimg         = 128<<10,  # Report status every N training images. None = disable.
    snapshot_nimg       = 8<<20,    # Save network snapshot every N training images. None = disable.
    checkpoint_nimg     = 128<<20,  # Save state checkpoint every N training images. None = disable.

    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    force_finite        = True,     # Get rid of NaN/Inf gradients before feeding them to the optimizer.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    pretrained_pkl      = './pretrained_model/edm2-img512-xs-2147483-0.135.pkl',
    device              = None,
):
    # Initialize.
    prev_status_time = time.time()
    misc.set_random_seed(seed)
    if device is None:
        device = dist.get_device()
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Validate batch size.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    assert total_nimg % batch_size == 0
    assert slice_nimg is None or slice_nimg % batch_size == 0
    assert status_nimg is None or status_nimg % batch_size == 0
    assert snapshot_nimg is None or (snapshot_nimg % batch_size == 0 and snapshot_nimg % 1024 == 0)
    assert checkpoint_nimg is None or (checkpoint_nimg % batch_size == 0 and checkpoint_nimg % 1024 == 0)

    use_cpu_gloo_grad_sync = _should_use_cpu_gloo_grad_sync(device)
    cpu_sync_group = torch.distributed.new_group(backend='gloo') if use_cpu_gloo_grad_sync else None
    if use_cpu_gloo_grad_sync:
        training_stats.init_multiprocessing(rank=dist.get_rank(), sync_device=torch.device('cpu'), sync_group=cpu_sync_group)

    # Setup dataset, encoder, and network.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    ref_image,ref_mask,ref_seg = dataset_obj[1]['img_latent'],dataset_obj[1]['mask_latent'],dataset_obj[1]['seg']
    dist.print0('Setting up encoder...')
    encoder_img = dnnlib.util.construct_class_by_name(**encoder_img_kwargs)
    ref_image = encoder_img.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    encoder_msk = dnnlib.util.construct_class_by_name(**encoder_msk_kwargs)
    ref_mask = encoder_msk.encode_latents(torch.as_tensor(ref_mask).to(device).unsqueeze(0))
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1]*2, label_dim=1)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().requires_grad_(True).to(device)

    # Print network summary.
    if dist.get_rank() == 0:
        net.eval()
        misc.print_module_summary(net, [
            torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device),
            torch.ones([batch_gpu], device=device),
            torch.zeros([batch_gpu, net.label_dim,ref_seg.shape[-1],ref_seg.shape[-1]], device=device),
        ], max_nesting=2)
        net.train()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    dist.barrier()

    # Setup training state.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)
    ddp_kwargs = {}
    if device.type == 'cuda':
        ddp_kwargs.update(device_ids=[device.index], output_device=device.index)
    if dist.get_world_size() > 1:
        ddp_kwargs.update(init_sync=False)
    ddp = torch.nn.parallel.DistributedDataParallel(net, **ddp_kwargs)
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    ema = dnnlib.util.construct_class_by_name(net=net, **ema_kwargs) if ema_kwargs is not None else None

    # Load previous checkpoint and decide how long to train.
    checkpoint = dist.CheckpointIO(state=state, net=net, loss_fn=loss_fn, optimizer=optimizer, ema=ema)
    pretrained_lag = checkpoint.load_latest(run_dir)
    if pretrained_lag is None:
        _load_image_warmstart(net=net, ema=ema, pretrained_pkl=pretrained_pkl)

    stop_at_nimg = total_nimg
    if slice_nimg is not None:
        granularity = checkpoint_nimg if checkpoint_nimg is not None else snapshot_nimg if snapshot_nimg is not None else batch_size
        slice_end_nimg = (state.cur_nimg + slice_nimg) // granularity * granularity # round down
        stop_at_nimg = min(stop_at_nimg, slice_end_nimg)
    assert stop_at_nimg > state.cur_nimg
    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {stop_at_nimg // 1000} kimg:')
    dist.print0()

    # Main training loop.
    misc.set_random_seed(seed, dist.get_rank())
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    stats_jsonl = None
    while True:
        done = (state.cur_nimg >= stop_at_nimg)

        # Report status.
        if status_nimg is not None and (done or state.cur_nimg % status_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            dist.print0(' '.join(['Status:',
                'kimg',         f"{training_stats.report0('Progress/kimg',                              state.cur_nimg / 1e3):<9.1f}",
                'time',         f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec',   state.total_elapsed_time)):<12s}",
                'sec/tick',     f"{training_stats.report0('Timing/sec_per_tick',                        cur_time - prev_status_time):<8.2f}",
                'sec/kimg',     f"{training_stats.report0('Timing/sec_per_kimg',                        cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3):<7.3f}",
                'maintenance',  f"{training_stats.report0('Timing/maintenance_sec',                     cur_time - prev_status_time - cumulative_training_time):<7.2f}",
                'cpumem',       f"{training_stats.report0('Resources/cpu_mem_gb',                       cpu_memory_usage / 2**30):<6.2f}",
                'gpumem',       f"{training_stats.report0('Resources/peak_gpu_mem_gb',                  torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
                'reserved',     f"{training_stats.report0('Resources/peak_gpu_mem_reserved_gb',         torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
            ]))
            cumulative_training_time = 0
            prev_status_nimg = state.cur_nimg
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()

            # Flush training stats.
            training_stats.default_collector.update()
            if dist.get_rank() == 0:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                fmt = {'Progress/tick': '%.0f', 'Progress/kimg': '%.3f', 'timestamp': '%.3f'}
                items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()

            # Update progress and check for abort.
            dist.update_progress(state.cur_nimg // 1000, stop_at_nimg // 1000)
            if state.cur_nimg == stop_at_nimg and state.cur_nimg < total_nimg:
                dist.request_suspend()
            if dist.should_stop() or dist.should_suspend():
                done = True

        # Save network snapshot.
        if snapshot_nimg is not None and state.cur_nimg % snapshot_nimg == 0 and (state.cur_nimg != start_nimg or start_nimg == 0) and dist.get_rank() == 0:
            ema_list = ema.get() if ema is not None else optimizer.get_ema(net) if hasattr(optimizer, 'get_ema') else net
            ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
            for ema_net, ema_suffix in ema_list:
                data = dnnlib.EasyDict(encoder_img=encoder_img, encoder_msk=encoder_msk, dataset_kwargs=dataset_kwargs, loss_fn=loss_fn)
                data.ema = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
                fname = f'network-snapshot-{state.cur_nimg//1000:07d}{ema_suffix}.pkl'
                dist.print0(f'Saving {fname} ... ', end='', flush=True)
                with open(os.path.join(run_dir, fname), 'wb') as f:
                    pickle.dump(data, f)
                dist.print0('done')
                del data # conserve memory

        # Save state checkpoint.
        if checkpoint_nimg is not None and (done or state.cur_nimg % checkpoint_nimg == 0) and state.cur_nimg != start_nimg:
            checkpoint.save(os.path.join(run_dir, f'training-state-{state.cur_nimg//1000:07d}.pt'))
            if use_cpu_gloo_grad_sync:
                _check_module_consistency_via_cpu(net, cpu_sync_group)
            else:
                misc.check_ddp_consistency(net)

        # Done?
        if done:
            break 

        # Evaluate loss and accumulate gradients.
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nimg)
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            sync_context = ddp.no_sync() if use_cpu_gloo_grad_sync else misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1))
            with sync_context:
                batch_sample= next(dataset_iterator)
                images, masks, segs = batch_sample['img_latent'],batch_sample['mask_latent'],batch_sample['seg']   
                images = encoder_img.encode_latents(images.to(device))
                masks = encoder_msk.encode_latents(masks.to(device))
                inputs = torch.cat([images,masks],dim=1)
                # loss = loss_fn(net=ddp, images=inputs, labels=segs.to(device))
                loss = loss_fn(net=ddp, images=inputs, labels=None)

                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()
        if use_cpu_gloo_grad_sync:
            _sync_module_grads_via_cpu(net, cpu_sync_group)

        # Run optimizer and update weights.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        training_stats.report('Loss/learning_rate', lr)
        for g in optimizer.param_groups:
            g['lr'] = lr
        if force_finite:
            for param in net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        optimizer.step()

        # Update EMA and training state.
        state.cur_nimg += batch_size
        if ema is not None:
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time

#----------------------------------------------------------------------------
