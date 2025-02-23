import os
import queue
import threading

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import imgproc

class TrainValidImageDataset(Dataset):
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): High resolution image size.
        upscale_factor (int): Image up scale factor.
        mode (str): Data set loading method, the training data set is for data enhancement, and the verification data set is not for data enhancement.
    """

    def __init__(self, image_dir: str, image_size: int, upscale_factor: int, mode: str) -> None:
        super(TrainValidImageDataset, self).__init__()
        # Get all image file names in folder
        self.image_file_names = [os.path.join(image_dir, image_file_name) for image_file_name in os.listdir(image_dir)]
        # Specify the high-resolution image size, with equal length and width
        self.image_size = image_size
        # How many times the high-resolution image is the low-resolution image
        self.upscale_factor = upscale_factor
        # Load training dataset or test dataset
        self.mode = mode

    def __getitem__(self, batch_index: int) -> dict[str, torch.Tensor]:
        # Read a batch of image data
        image = cv2.imread(self.image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        # Image processing operations
        if self.mode == "Train":
            hr_image = imgproc.random_crop(image, self.image_size)
        elif self.mode == "Valid":
            hr_image = imgproc.center_crop(image, self.image_size)
        else:
            raise ValueError("Unsupported data processing model, please use `Train` or `Valid`.")

        lr_image = imgproc.image_resize(hr_image, 1 / self.upscale_factor)
        lr_image = imgproc.image_resize(lr_image, self.upscale_factor)

        # Only extract the image data of the Y channel
        lr_y_image = imgproc.bgr2ycbcr(lr_image, only_use_y_channel=True)
        hr_y_image = imgproc.bgr2ycbcr(hr_image, only_use_y_channel=True)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=False)
        hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=False)

        return {"lr": lr_y_tensor, "hr": hr_y_tensor}

    def __len__(self) -> int:
        return len(self.image_file_names)