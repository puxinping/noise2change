import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, binary_dilation

# --- Class color/label mappings ------------------------------------------------
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


def make_rgb(label_map: np.ndarray, grey_codes: dict = class_grey_oem, rgb_codes: dict = class_rgb_oem) -> np.ndarray:
    """Convert a grey label map (H, W) to an RGB image (H, W, 3).

    This function is a pure-NumPy utility for visualization.
    """
    out = np.zeros(shape=label_map.shape + (3,), dtype=np.uint8)
    for name, grey in grey_codes.items():
        rgb = rgb_codes[name]
        out[label_map == grey, 0] = rgb[0]
        out[label_map == grey, 1] = rgb[1]
        out[label_map == grey, 2] = rgb[2]
    return out


def rgb_to_class_mask_np(rgb_image: np.ndarray, class_rgb: dict = class_rgb_oem, class_grey: dict = class_grey_oem) -> np.ndarray:
    """Non-differentiable, NumPy implementation that maps every pixel to nearest class RGB.

    Args:
        rgb_image: (H, W, 3) uint8 or float image
    Returns:
        class_mask: (H, W) int array of grey labels
    """
    class_names = list(class_rgb.keys())
    class_rgb_arr = np.array(list(class_rgb.values()), dtype=np.float32)  # (C,3)
    h, w, _ = rgb_image.shape
    pixels = rgb_image.reshape(-1, 3).astype(np.float32)
    # distances: (N, C)
    dists = np.linalg.norm(pixels[:, None, :] - class_rgb_arr[None, :, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    class_mask_flat = np.array([class_grey[class_names[i]] for i in nearest], dtype=np.int64)
    return class_mask_flat.reshape(h, w)


def rgb_to_class_mask_torch(rgb_image: torch.Tensor, class_rgb: dict = class_rgb_oem, class_grey: dict = class_grey_oem, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
    """Differentiable mapping from RGB image to class mask using (Gumbel-)Softmax.

    Args:
        rgb_image: (H, W, 3) torch.Tensor float32
        tau: temperature for gumbel_softmax
        hard: if True, returns hard one-hot selection (still differentiable in PyTorch)
    Returns:
        class_mask: (H, W) float tensor with class grey-values (may be non-integer if soft)
    """
    device = rgb_image.device
    class_rgb_arr = torch.tensor(list(class_rgb.values()), dtype=torch.float32, device=device)  # (C,3)
    class_grey_vals = torch.tensor(list(class_grey.values()), dtype=torch.float32, device=device)  # (C,)

    h, w, _ = rgb_image.shape
    pixels = rgb_image.view(-1, 3)  # (N,3)
    # distances (N,C)
    dists = torch.cdist(pixels, class_rgb_arr)  # uses Euclidean
    probs = F.gumbel_softmax(-dists, tau=tau, hard=hard, dim=1)  # (N,C)
    class_mask_flat = probs.matmul(class_grey_vals)  # (N,)
    return class_mask_flat.view(h, w)


def morphological_processing(class_mask: np.ndarray, operation: str = 'dilation', kernel_size: int = 3) -> np.ndarray:
    """Simple morphological post-processing using SciPy.

    Args:
        class_mask: (H, W) int array
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=bool)
    if operation == 'erosion':
        return binary_erosion(class_mask, structure=kernel).astype(np.uint8)
    if operation == 'dilation':
        return binary_dilation(class_mask, structure=kernel).astype(np.uint8)
    raise ValueError("operation must be 'erosion' or 'dilation'")


def fill_holes_with_voting(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Fill zero-valued holes in a class mask by neighborhood voting (PyTorch version).

    This keeps everything on the same device as `mask`.
    """
    if not isinstance(mask, torch.Tensor):
        raise TypeError('mask must be a torch.Tensor')
    filled = mask.clone()
    H, W = mask.shape
    pad = kernel_size // 2
    for i in range(H):
        i0 = max(0, i - pad)
        i1 = min(H, i + pad + 1)
        for j in range(W):
            if mask[i, j] == 0:
                j0 = max(0, j - pad)
                j1 = min(W, j + pad + 1)
                neighborhood = mask[i0:i1, j0:j1]
                neighborhood_nonzero = neighborhood[neighborhood != 0]
                if neighborhood_nonzero.numel() > 0:
                    vals, counts = torch.unique(neighborhood_nonzero, return_counts=True)
                    filled[i, j] = vals[torch.argmax(counts)]
                else:
                    filled[i, j] = 0
    return filled


def convert_non_integer_to_zero(mask: torch.Tensor) -> torch.Tensor:
    """Set non-integer values in a (float) mask to zero.

    Useful after differentiable soft assignments.
    """
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask)
    is_integer = (mask == mask.floor())
    return mask * is_integer


if __name__ == '__main__':
    # small demo (requires torchvision/Pillow and torch)
    img_path = '/data/EDM2/images/demo/demo/000000.png'
    im = Image.open(img_path).convert('RGB')
    rgb = np.array(im)
    # CPU NumPy mapping
    mask_np = rgb_to_class_mask_np(rgb)
    rgb_vis = make_rgb(mask_np)
    Image.fromarray(rgb_vis).show()
