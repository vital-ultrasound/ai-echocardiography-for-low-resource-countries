from time import time
from typing import Tuple

import cv2 as cv
import numpy as np
import torch


def ToImageTensor(image_np_array: np.ndarray, nch: int = 1 or None) -> torch.Tensor:
    """
    change_video_shape "numpy image: H x W x C" to torch image: C x H x W

    Arguments:
        image_np_array: Numpy array of image frame

    Return:
        torch.Tensor image in the form of chs_h_w
    """

    if nch == 3:
        image_np_array_ = cv.cvtColor(image_np_array, cv.COLOR_BGR2RGB)
        frame_torch = torch.as_tensor(image_np_array_, dtype=torch.float32) #frame_torch_h_w_c
        frame_torch = torch.movedim(frame_torch, -1, 0) #frame_torch_c_h_w

    elif nch == 1:
        image_np_array_ = cv.cvtColor(image_np_array, cv.COLOR_BGR2GRAY)
        frame_torch = torch.as_tensor(image_np_array_, dtype=torch.float32) #frame_torch_h_w
        frame_torch = frame_torch.unsqueeze(0) # Fake batch dimension to be "C,H,W"

    else:
        print(f'Should be 1 or 3 channels')

    return frame_torch

def to_grayscale(image_np_array: np.ndarray, color_th: int = 1) -> np.ndarray:
    """
    Convert BGR to grayscale.
    NOTE: This is an expensive operation because of the std deviation and conditions to check
        colour threshold!

    Arguments:
        image_np_array: Numpy array of image frame

    Return:
        gray_image_np_array: Numpy array of image frame
    """
    color_th = 1
    nongray = np.std(image_np_array, axis=2)
    gray_image_np_array = cv.cvtColor(image_np_array, cv.COLOR_BGR2GRAY)
    gray_image_np_array[nongray > color_th] = 0

    return gray_image_np_array

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
