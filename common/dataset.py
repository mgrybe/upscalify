import h5py
import numpy as np
from PIL import Image
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms

def _to_pil_rgb(arr: np.ndarray) -> Image.Image:
    """Accept HxW or HxWxC uint8; return PIL RGB."""
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    # Use automatic mode inference; passing `mode` is deprecated in Pillow 13
    return Image.fromarray(arr.astype(np.uint8))

def _modcrop_pil(img: Image.Image, scale: int) -> Image.Image:
    w, h = img.size
    return img.crop((0, 0, w - (w % scale), h - (h % scale)))


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.uint8).copy()
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    #t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32) / 255.0
    return t

def mixup(lq, gt, alpha=1.2):
    if random.random() < 0.5:
        return lq, gt

    v = np.random.beta(alpha, alpha)
    r_index = torch.randperm(lq.size(0)).to(gt.device)

    lq = v * lq + (1 - v) * lq[r_index, :]
    gt = v * gt + (1 - v) * gt[r_index, :]
    return lq, gt

def shuffle_channels(lq, gt):
    if random.random() < 0.5:
        return lq, gt

    perm = np.random.permutation(3)
    lq = lq[perm, :, :]
    gt = gt[perm, :, :]

    return lq, gt

# class RandomChannelShuffle:
#     """Apply single transformation randomly picked from a list. This transform does not support torchscript."""
#
#     def __init__(self, p=0.5):
#         self.p = p
#
#     def __call__(self, x):
#         if random.random() < self.p:
#             return x
#
#         perm = np.random.permutation(3)
#         t = x[perm, :, :]
#
#         return t
#
#     def __repr__(self) -> str:
#         return f"{super().__repr__()}(p={self.p})"

class TrainH5Dataset(Dataset):
    """
    Minimal, SwinIR-friendly HR-only H5 training dataset.
    - Reads HR from H5, generates LR via bicubic.
    - Augment with random flips and 90° rotations.
    - Random HR crop of size `crop_size` (so LR is `crop_size//scale_factor`).
    """
    def __init__(self,
                 h5_file: str,
                 crop_size: int = 96,
                 scale_factor: int = 4,
                 augment: bool = True,
                 modcrop: bool = True,
                 channel_shuffle: bool = False):
        super().__init__()
        self.h5_file = h5_file
        self.crop_size = int(crop_size)              # HR crop size
        self.scale = int(scale_factor)
        self.augment = augment
        self.modcrop = modcrop
        self.channel_shuffle = channel_shuffle

        # Compose simple transforms (apply to HR; LR is derived after)
        aug_ops = []
        if self.augment:
            aug_ops += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomChoice([
                    transforms.Lambda(lambda x: x),              # 0°
                    transforms.Lambda(lambda x: x.rotate(90)),
                    transforms.Lambda(lambda x: x.rotate(180)),
                    transforms.Lambda(lambda x: x.rotate(270)),
                ])
            ]
        # Crop after aug to keep sizes tidy; pad if needed for small images.
        aug_ops += [transforms.RandomCrop(self.crop_size, pad_if_needed=True)]
        self.hr_transform = transforms.Compose(aug_ops)

        # LR is just HR cropped then bicubic shrink by the scale
        self._lr_resize = transforms.Resize(self.crop_size // self.scale, interpolation=Image.BICUBIC)

        # We keep it simple and open the file on each __getitem__ (robust for num_workers>0).
        with h5py.File(self.h5_file, 'r') as f:
            self._keys = sorted(list(f.keys()), key=lambda k: int(k) if k.isdigit() else k)

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, idx):
        key = self._keys[idx]
        with h5py.File(self.h5_file, 'r') as f:
            arr = np.array(f[key])  # HxW or HxWxC, uint8
        hr = _to_pil_rgb(arr)

        # Modcrop to multiple of scale (keeps alignment exact)
        if self.modcrop:
            hr = _modcrop_pil(hr, self.scale)

        # Augment + random HR crop
        hr = self.hr_transform(hr)

        # Generate LR from HR crop
        lr = self._lr_resize(hr)

        # To tensors in [0,1]
        lr_t = _to_tensor(lr)
        hr_t = _to_tensor(hr)

        if self.channel_shuffle:
            lr_t, hr_t = shuffle_channels(lr_t, hr_t)

        return lr_t, hr_t


class EvalH5Dataset(Dataset):
    """
    Evaluation/Test dataset for SR that opens an HDF5 file
    and selects one group/key inside (e.g. "set5", "set14").

    Example:
        >>> ds = EvalH5Dataset('./data/superres-test.h5', key='set5', scale_factor=4)
        >>> lr, hr = ds[0]
    """
    def __init__(self, h5_file: str, key: str, scale_factor: int = 4, modcrop: bool = True):
        super().__init__()
        self.h5_file = h5_file
        self.key = key
        self.scale = int(scale_factor)
        self.modcrop = modcrop

        # preload keys once
        with h5py.File(self.h5_file, 'r') as f:
            self._group = f[self.key]
            self._keys = sorted(list(self._group.keys()), key=lambda k: int(k) if k.isdigit() else k)

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            arr = np.array(f[self.key][self._keys[idx]])
        hr = _to_pil_rgb(arr)

        if self.modcrop:
            hr = _modcrop_pil(hr, self.scale)

        w, h = hr.size
        lr = hr.resize((w // self.scale, h // self.scale), Image.BICUBIC)

        return _to_tensor(lr), _to_tensor(hr)
