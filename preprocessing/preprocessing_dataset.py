
import os
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import rasterio
import torch
from scipy.ndimage import label
import glob
from torch_utils. mask_vis import make_rgb
def load_multiband(path: str):
    src = rasterio.open(path, "r")
    return (np.moveaxis(src.read(), 0, -1)).astype(np.uint8)


def load_grayscale(path: str):
    src = rasterio.open(path, "r")
    return (src.read(1)).astype(np.uint8)


class CustomImageMaskDataset(Dataset):
    def __init__(self, root_dir, transform=None, conditioning_image_transforms=None):
        self.root_dir = root_dir
        self.transform = transform
        self.conditioning_image_transforms = conditioning_image_transforms
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale
        self.image_fps = [f for ext in ('*.tif', '*.png') for f in glob.glob(os.path.join(root_dir, 'img', ext))][:100]
        self.msks_fps = [f.replace("img", "msk") for f in self.image_fps]
        self.ignore_label = 0

    def __len__(self):
        return len(self.image_fps)

    def calculate_connected_components_and_ratios(self, mask, num_classes):
        total_pixels = mask.numel()
        counts = []
        ratios = []
        for i in range(num_classes):
            binary_mask = (mask == i).numpy().astype(np.uint8)
            num_features = label(binary_mask)[1]  # 直接获取连通区域数量
            counts.append(num_features)
            ratios.append(binary_mask.sum() / total_pixels if total_pixels > 0 else 0)
        label_info = torch.stack([torch.tensor(counts), torch.tensor(ratios)],dim=1)
        return label_info

    def __getitem__(self, idx): #19
        image = self.load_multiband(self.image_fps[idx])
        mask = self.load_multiband(self.msks_fps[idx])
        filename = os.path.splitext(os.path.basename(self.image_fps[idx]))[0]
        y = self.calculate_connected_components_and_ratios(torch.from_numpy(mask),9)
        image = torch.from_numpy(np.array(image)).permute(2,0,1)
        mask_rgb = make_rgb(mask[:,:,0])
        mask_rgb = torch.from_numpy(np.array(mask_rgb)).permute(2,0,1)
        return image, mask_rgb, y, mask[:,:,0],filename
