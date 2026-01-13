import rasterio
import numpy as np
from PIL import Image

def load_multiband(path: str):
    with rasterio.open(path, "r") as src:
        return (np.moveaxis(src.read(), 0, -1)).astype(np.uint8)

def load_grayscale(path: str):
    with rasterio.open(path, "r") as src:
        return (src.read(1)).astype(np.uint8)

def save_image(array, path=None, mode='RGB'):
    img = Image.fromarray(array, mode)
    if path: 
        img.save(path)
    return img  