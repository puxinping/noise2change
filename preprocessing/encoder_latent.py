import os
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
import tqdm
import cv2
import dnnlib
from preprocessing.preprocessing_dataset import CustomImageMaskDataset



def center_crop_arr(pil_image, image_size):
    """Center-crop PIL image to square of size `image_size`.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def main(args):
    assert torch.cuda.is_available(), "Requires at least one GPU."

    # Initialize process group (torchrun must be used to spawn processes)
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    if rank == 0:
        os.makedirs(args.features_path, exist_ok=True)
        os.makedirs(os.path.join(args.features_path, 'img_latent'), exist_ok=True)
        os.makedirs(os.path.join(args.features_path, 'msk_latent'), exist_ok=True)
        os.makedirs(os.path.join(args.features_path, 'img'), exist_ok=True)
        os.makedirs(os.path.join(args.features_path, 'msk'), exist_ok=True)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    conditioning_image_transforms = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    dataset = CustomImageMaskDataset(args.data_path, transform=transform,
                                     conditioning_image_transforms=conditioning_image_transforms)

    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=False, seed=args.global_seed)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # Create encoder instances (constructed via dnnlib factory)
    encoder_img = dnnlib.util.construct_class_by_name(class_name='training.encoders_image.StabilityVAEEncoder')
    encoder_msk = dnnlib.util.construct_class_by_name(class_name='training.encoders_mask.StabilityVAEEncoder')
    encoder_img.init(torch.device(device))
    encoder_msk.init(torch.device(device))

    for image, masks_rgb, y, mask, name in tqdm.tqdm(loader):
        image = image.to(device, dtype=torch.float32)
        masks_rgb = masks_rgb.to(device, dtype=torch.float32)
        with torch.no_grad():
            mean_std_img = encoder_img.encode_pixels(image).cpu()
            mean_std_msk = encoder_msk.encode_pixels(masks_rgb).cpu()
            # Save latents as numpy files
            np.save(os.path.join(args.features_path, 'img_latent', f"{name[0]}.npy"), mean_std_img.numpy())
            np.save(os.path.join(args.features_path, 'msk_latent', f"{name[0]}.npy"), mean_std_msk.numpy())
            # Optional: Decode and save reconstructed images for verification
            # images = encoder_msk.encode_latents(mean_std_img.to(device=device,dtype=torch.float32))
            # images = encoder_msk.decode(images).permute(0, 2, 3, 1).cpu().numpy()[0]
            # images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(os.path.join(args.features_path, 'img', f"{name[0]}.tif"), images)
            # # mask visualization
            # mask = encoder_msk.encode_latents(mean_std_msk.to(device=device,dtype=torch.float32))
            # mask = encoder_msk.decode(mask).permute(0, 2, 3, 1).cpu().numpy()[0]
            # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(os.path.join(args.features_path, 'msk', f"{name[0]}.tif"), mask)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='../datasets')
    parser.add_argument("--features-path", type=str, default="../feature")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--global-batch-size", type=int, default=1)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--vae-model-path", type=str, default="./pretrained_model/sd-vae-ft-mse")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    main(args)