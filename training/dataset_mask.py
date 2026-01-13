# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import cv2
try:
    import pyspng
except ImportError:
    pyspng = None
import rasterio
#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        use_labels  = True,     # Enable conditioning labels? False = label dimension is zero.
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self._raw_shape[1:]
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self): # [CHW]
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = anything goes.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in supported_ext)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        ext = self._file_ext(fname)
        with self._open_file(fname) as f:
            if ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------

# class CustomDataset(Dataset):
#     def __init__(self,
#         path,                   # Path to directory or zip.
#         resolution      = None, # Ensure specific resolution, None = anything goes.
#         **super_kwargs,         # Additional arguments for the Dataset base class.
#     ):
#         self._path = path
#         self._zipfile = None

#         if os.path.isdir(self._path):
#             self._type = 'dir'
#             self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
#         elif self._file_ext(self._path) == '.zip':
#             self._type = 'zip'
#             self._all_fnames = set(self._get_zipfile().namelist())
#         else:
#             raise IOError('Path must point to a directory or zip')

#         PIL.Image.init()
#         supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}
#         self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in supported_ext)
#         if len(self._image_fnames) == 0:
#             raise IOError('No image files found in the specified path')

#         name = os.path.splitext(os.path.basename(self._path))[0]
#         raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
#         if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
#             raise IOError('Image files do not match the specified resolution')
#         super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

#     @staticmethod
#     def _file_ext(fname):
#         return os.path.splitext(fname)[1].lower()

#     def _get_zipfile(self):
#         assert self._type == 'zip'
#         if self._zipfile is None:
#             self._zipfile = zipfile.ZipFile(self._path)
#         return self._zipfile

#     def _open_file(self, fname):
#         if self._type == 'dir':
#             return open(os.path.join(self._path, fname), 'rb')
#         if self._type == 'zip':
#             return self._get_zipfile().open(fname, 'r')
#         return None

#     def close(self):
#         try:
#             if self._zipfile is not None:
#                 self._zipfile.close()
#         finally:
#             self._zipfile = None

#     def __getstate__(self):
#         return dict(super().__getstate__(), _zipfile=None)

#     def _load_raw_image(self, raw_idx):
#         fname = self._image_fnames[raw_idx]
#         ext = self._file_ext(fname)
#         with self._open_file(fname) as f:
#             if ext == '.npy':
#                 image = np.load(f)
#                 image = image.reshape(-1, *image.shape[-2:])
#             elif ext == '.png' and pyspng is not None:
#                 image = pyspng.load(f.read())
#                 image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
#             else:
#                 image = np.array(PIL.Image.open(f))
#                 image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
#         return image

#     def _load_raw_mask(self, raw_idx):
#         fname = self._image_fnames[raw_idx]
#         ext = self._file_ext(fname)
#         with self._open_file(fname) as f:
#             if ext == '.npy':
#                 image = np.load(f)
#                 image = image.reshape(-1, *image.shape[-2:])
#             elif ext == '.png' and pyspng is not None:
#                 image = pyspng.load(f.read())
#                 image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
#             else:
#                 image = np.array(PIL.Image.open(f))
#                 image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
#         return image

#     def _load_raw_seg(self, raw_idx):
#         fname = self._image_fnames[raw_idx]
#         ext = self._file_ext(fname)
#         with self._open_file(fname) as f:
#             if ext == '.npy':
#                 image = np.load(f)
#                 image = image.reshape(-1, *image.shape[-2:])
#             elif ext == '.png' and pyspng is not None:
#                 image = pyspng.load(f.read())
#                 image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
#             else:
#                 image = np.array(PIL.Image.open(f))
#                 image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
#         return image

#     def _load_raw_label(self, raw_idx):
#         fname = self._image_fnames[raw_idx]
#         ext = self._file_ext(fname)
#         with self._open_file(fname) as f:
#             if ext == '.npy':
#                 image = np.load(f)
#                 image = image.reshape(-1, *image.shape[-2:])
#             elif ext == '.png' and pyspng is not None:
#                 image = pyspng.load(f.read())
#                 image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
#             else:
#                 image = np.array(PIL.Image.open(f))
#                 image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
#         return image


#     def __getitem__(self, idx):
#         raw_idx = self._raw_idx[idx]
#         image = self._load_raw_image(raw_idx)
#         mask = self._load_raw_mask(raw_idx)
#         seg = self._load_raw_seg(raw_idx)
#         image = self._load_raw_label(raw_idx)

#         assert isinstance(image, np.ndarray)
#         assert list(image.shape) == self._raw_shape[1:]
#         if self._xflip[idx]:
#             assert image.ndim == 3 # CHW
#             image = image[:, :, ::-1]
#         return image.copy(),image.copy(),mask.copy(),seg.copy(),self.get_label(idx)


class CustomDataset(Dataset):
    def __init__(self, path, use_labels=False):
        # features_dir, _features/_labels
        self._use_labels = use_labels
        L = os.listdir(path)
        print(f'---> Folders in {path}: {L}')
        for name in L:
            if name.endswith('msk'):
                self.mask_dir = os.path.join(path, name)
        self.mask_files = sorted(os.listdir(self.mask_dir))
        self._raw_shape = [len(self.mask_files)] + list(self.load_grayscale(os.path.join(self.mask_dir, self.mask_files[0])).shape)
    def __len__(self):
        return len(self.mask_files)

    def load_grayscale(self, path: str):
        src = rasterio.open(path, "r")
        return (src.read(1)).astype(np.uint8)

    def __getitem__(self, idx):
        mask_file = self.mask_files[idx]
        masks = self.load_grayscale(os.path.join(self.mask_dir, mask_file))
        masks = masks[::8, ::8]
        # masks = cv2.resize(masks, (128, 128)).astype(np.int32)
        if self._use_labels == True:
            unique_classes = np.unique(masks)
            num_to_select = np.random.randint(1, len(unique_classes) + 1)
            selected_classes = np.random.choice(unique_classes, size=max(1, num_to_select), replace=False)
            # 将掩膜图像中的像素值映射到对应的独热编码通道
            labels = np.zeros(9, dtype=np.int32)
            labels[selected_classes] = 1  # 存在的类别对应位置设置为 1
        else:
            labels = None
        return torch.from_numpy(masks),labels


    def sample_non_zero_class(self, labels):
        new_labels = torch.zeros(labels.size(0), dtype=torch.long)
        for i in range(labels.size(0)):
            non_zero_classes = (labels[i, :, 1] >= 0.15).nonzero(as_tuple=True)[0]
            if len(non_zero_classes) == 0:  # 如果没有符合 >= 0.2 的类别
                non_zero_classes = (labels[i, :, 1] > 0).nonzero(as_tuple=True)[0]
            if len(non_zero_classes) > 0:  # 随机选择非零类别
                new_labels[i] = non_zero_classes[torch.randint(0, len(non_zero_classes), (1,))]
            else:  # 如果全为零，随机选择一个类别
                new_labels[i] = torch.randint(0, labels.size(1), (1,))
        return new_labels

    # def sample_non_zero_class(self,labels):
    #     # 获取每行非零类别的索引，并随机选择
    #     non_zero_indices = (labels[:,:,1] >= 0.2)
    #     new_labels = torch.zeros(labels.size(0), dtype=torch.long)
    #     for i in range(labels.size(0)):
    #         non_zero_classes = non_zero_indices[i].nonzero(as_tuple=True)[0]
    #         if len(non_zero_classes) > 0:
    #             new_labels[i] = non_zero_classes[torch.randint(0,len(non_zero_classes), (1,))]
    #         else:
    #                 new_labels[i] = torch.randint(0, labels.size(1), (1,))

    #     return new_labels

    @property
    def image_shape(self): # [CHW]
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]



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
