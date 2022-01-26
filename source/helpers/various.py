import random
from pathlib import Path
from time import time
from typing import Tuple, List

import cv2 as cv
import numpy as np
import torch


def ToImageTensor(image_np_array: np.ndarray) -> torch.Tensor:
    """
    Torch image tensor as C x H x W

    Arguments:
        image_np_array: Numpy array of image frame

    Return:
        torch.Tensor image in the form of chs_h_w
    """

    # frame_torch = torch.as_tensor(image_np_array, dtype=torch.float32)
    frame_torch = torch.from_numpy(image_np_array).float()
    frame_torch = frame_torch.unsqueeze(0)  # Fake batch dimension to be "C,H,W"

    return frame_torch


def to_grayscale(image_np_array: np.ndarray, color_th: bool = False or None) -> np.ndarray:
    """
    Convert BGR to grayscale.
    NOTE: This is an expensive operation because of the std deviation and conditions to check
        colour threshold!

    Arguments:
        image_np_array: Numpy array of image frame

    Return:
        gray_image_np_array: Numpy array of image frame
    """
    if color_th == True:
        color_th_ = 1
        nongray = np.std(image_np_array, axis=2)
        gray_image_np_array = cv.cvtColor(image_np_array, cv.COLOR_BGR2GRAY)
        gray_image_np_array[nongray > color_th_] = 0
    else:
        gray_image_np_array = cv.cvtColor(image_np_array,
                                          cv.COLOR_BGR2GRAY)  # default is unit8 but you can consider .astype(np.float64)

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


def cropped_frame(image_frame_array_3ch: np.ndarray, crop_bounds: List) -> np.ndarray:
    """
    Hard crop of US image with bounds: (start_x, start_y, width, height)
    """
    cropped_image_frame_ = image_frame_array_3ch[
                           int(crop_bounds['start_y']):int(crop_bounds['start_y'] + crop_bounds['height']),
                           int(crop_bounds['start_x']):int(crop_bounds['start_x'] + crop_bounds['width'])]

    return cropped_image_frame_


def masks_us_image(image_frame_array_1ch: np.ndarray) -> np.ndarray:
    """
    Hard mask pixels outside of scanning sector
    """
    mask = np.zeros_like(image_frame_array_1ch)

    x_data = np.array([1050, 1532, 1428, 1310, 1188, 1053, 938, 835, 747, 645, 568, 1041])
    y_data = np.array([133, 759, 830, 879, 911, 922, 915, 890, 862, 812, 760, 133])
    scan_arc_mask_v01 = np.vstack((x_data, y_data)).astype(np.int32).T

    caliper_scale_mask = np.array([(1770, 120), (1810, 120), (1810, 930), (1770, 930)])

    cv.fillPoly(mask, [scan_arc_mask_v01],
                (255, 255, 0))
    maskedImage = cv.bitwise_and(image_frame_array_1ch, image_frame_array_1ch, mask=mask)

    return maskedImage


def write_list_to_txtfile(list: List, filename: str, files_path: str) -> None:
    """

    Write a txt file from a list

    Arguments:
        list: list of filenames
        filename: string of the name where txt will be saved (e.g.: video_list_train.txt)
        files_path: path where you will save txt files

    Return:
        None

    """
    textfile = open('{}{}'.format(files_path, filename), "w")
    for element in list:
        textfile.write(element + "\n")


def split_train_validate_sets(echodataset_path: str, data_list_output_path: str, ntraining: float) -> None:
    """

    Split paths to train and validate sets

    Arguments:
        echodataset_path: path of the video and annotation files

        data_list_output_path: path where text files containing lists of data (train/validate videos
                               and annotations)  are written

        ntraining: percentage of data used for training from 0.0 to 1.0

    Return:
        None

    """
    nvalidation = 1.0 - ntraining

    all_videos_file = 'video_list_full.txt'
    all_labels_file = 'annotation_list_full.txt'

    videolist = '{}{}'.format(data_list_output_path, all_videos_file)
    labellist = '{}{}'.format(data_list_output_path, all_labels_file)

    ## List all files
    result = list(Path(echodataset_path).rglob("*echo*.[mM][pP][4]"))
    with open(videolist, 'w') as f:
        for fn in result:
            fn_nopath = str(fn).replace(echodataset_path, '')
            f.write(fn_nopath + '\n')

    result = list(Path(echodataset_path).rglob("*4CV.[jJ][sS][oO][nN]"))
    with open(labellist, 'w') as f:
        for fn in result:
            fn_nopath = str(fn).replace(echodataset_path, '')
            f.write(fn_nopath + '\n')

    ## Load filenames into list
    video_filenames = [line.strip() for line in open(videolist)]
    label_filenames = [line.strip() for line in open(labellist)]

    ## Randomly shuffle lists
    c = list(zip(video_filenames, label_filenames))
    random.shuffle(c)
    video_filenames, label_filenames = zip(*c)

    ## Split and save txt files
    N = len(video_filenames)
    video_filenames_t = video_filenames[:int(N * ntraining)]
    label_filenames_t = label_filenames[:int(N * ntraining)]
    video_filenames_v = video_filenames[int(N * ntraining):]
    label_filenames_v = label_filenames[int(N * ntraining):]

    write_list_to_txtfile(video_filenames_t, 'video_list_train.txt', data_list_output_path)
    write_list_to_txtfile(label_filenames_t, 'annotation_list_train.txt', data_list_output_path)
    write_list_to_txtfile(video_filenames_v, 'video_list_validate.txt', data_list_output_path)
    write_list_to_txtfile(label_filenames_v, 'annotation_list_validate.txt', data_list_output_path)
