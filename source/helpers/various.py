from time import time
from typing import Tuple

import cv2 as cv
import numpy as np
import torch


def ToTensor(image_np_array: np.ndarray) -> torch.Tensor:
    """
    change_video_shape "numpy image: H x W x C" to torch image: C x H x W

    Arguments:
        image_np_array: Numpy array of image frame

    Return:
        torch.Tensor image in the form of chs_h_w
    """

    torch_frame_hwc = torch.as_tensor(image_np_array) # torch.uint8
    torch_frame_cwh = torch.movedim(torch_frame_hwc, -1, 0)

    return torch_frame_cwh



def show_torch_tensor(tensor: torch.Tensor) -> None:
    """
    Display a  torch.Tensor image on screen
    :param tensor: image to visualise, of size (h, w, 1/3)
    :type tensor:  torch.Tensor
    :param zoom: zoom factor
    :type zoom: float
    """

    cv.imshow('a', tensor.permute(1, 2, 0).numpy())


def timer_func_decorator(func):
    """
    This function shows the execution time of
    the function object passed
    """

    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


def msec_to_timestamp(current_timestamp: float) -> Tuple[float]:
    """
    Convert millisecond variable to a timestamp variable with the format minutes, seconds and milliseconds
    """
    minutes = int(current_timestamp / 1000 / 60)
    seconds = int(np.floor(current_timestamp / 1000) % 60)
    ms = current_timestamp - np.floor(current_timestamp / 1000) * 1000

    return minutes, seconds, '{:.3f}'.format(ms), '{:02d}:{:02d}:{:.3f}'.format(minutes, seconds, ms)
