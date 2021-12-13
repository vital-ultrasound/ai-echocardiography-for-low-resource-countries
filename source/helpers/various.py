from time import time
from typing import Tuple

import cv2 as cv
import numpy as np
import torch


def change_video_tensor_shape(image_np_array: np.ndarray, nch: int = 3 or None) -> torch.Tensor:
    """
    change_video_shape to Timestamp in msec, Clip_number, Channels, Height, Width
    TODO:
        * ChangeVideoShape to "CTHW" (Channels, Time, Height, Width)
        * make use of torch.from_numpy(image_frame_array_3ch_i).float().cuda()

    Arguments:
        image_np_array: Numpy array of image frame
        nch: Number of channels with one channel as default or None if not specified

    Return:
        torch.Tensor in the form of idx_chs_h_w
    """

    if nch == 3:
        image_np_array_ = cv.cvtColor(image_np_array, cv.COLOR_BGR2RGB)
    elif nch == 1:
        image_np_array_ = cv.cvtColor(image_np_array, cv.COLOR_BGR2GRAY)
    else:
        print(f'Should be 1 or 3 channels')

    torch_frame_h_w_chs = torch.as_tensor(image_np_array_)
    torch_frame_idx_chs_h_w = torch.movedim(torch_frame_h_w_chs, -1, 0)

    return torch_frame_idx_chs_h_w


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
