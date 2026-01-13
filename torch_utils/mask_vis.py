import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path

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

class_grey_oem = {
    "unknown": 0,
    "Bareland": 1,
    "Grass": 2,
    "Pavement": 3,
    "Road": 4,
    "Tree": 5,
    "Water": 6,
    "Cropland": 7,
    "buildings": 8,
}


class_grey_oem = {k: i for i, k in enumerate(class_rgb_oem)}


def rgb_to_class_mask(rgb_image, class_rgb_oem=class_rgb_oem, class_grey_oem=class_grey_oem):
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



# make_rgb 函数
def make_rgb(a, grey_codes: dict = class_grey_oem, rgb_codes: dict = class_rgb_oem):
    """Convert a grey label map to an RGB color-coded image.

    Args:
        a (numpy array): semantic label (H x W)
        grey_codes (dict): dict of label code to grey level
        rgb_codes (dict): dict of label to RGB color

    Returns:
        np.array: RGB color-coded semantic label map
    """
    out = np.zeros(shape=a.shape + (3,), dtype="uint8")
    for k, v in grey_codes.items():
        out[a == v, 0] = rgb_codes[k][0]
        out[a == v, 1] = rgb_codes[k][1]
        out[a == v, 2] = rgb_codes[k][2]
    return out

def convert_labels_to_rgb(labels_dir, output_dir, grey_codes=class_grey_oem, rgb_codes=class_rgb_oem):
    label_files = [f for f in Path(labels_dir).rglob("*.tif")][:10]  # 根据需要修改文件类型，如 .tif
    os.makedirs(output_dir, exist_ok=True)
    for label_file in label_files:
        label_img = np.array(Image.open(label_file))
        label_rgb = make_rgb(label_img, grey_codes, rgb_codes)
        output_file = os.path.join(output_dir, label_file.name)
        Image.fromarray(label_rgb).save(output_file)
        print(f"已保存: {output_file}")

if __name__ == '__main__':
    labels_dir = "./datasets/msk"
    output_dir = "./preprocessing/demo"

    convert_labels_to_rgb(labels_dir, output_dir)