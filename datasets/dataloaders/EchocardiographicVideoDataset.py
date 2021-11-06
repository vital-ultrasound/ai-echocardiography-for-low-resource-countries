

import torch
import torch.utils.data as Data
import os
import json
from typing import Tuple, List
import cv2 as cv
import numpy as np
#import SimpleITK as sitk
#import csv
#from utils import utils
#import itertools
#import random

S2MS = 1000

def convert_sec_to_min_sec_ms(timestamp_in_secs: float) -> Tuple[float]:
    """
    Convert seconds to the format of minutes, seconds and milliseconds.
    Temporal segments in VGG I A are in seconds.
    References: https://gitlab.com/vgg/via/blob/master/via-3.x.y/CodeDoc.md
    """
    day = timestamp_in_secs // (24 * 3600)
    timestamp_in_secs = timestamp_in_secs % (24 * 3600)
    hour = timestamp_in_secs // 3600
    timestamp_in_secs %= 3600
    minutes = timestamp_in_secs // 60
    timestamp_in_secs %= 60
    seconds = timestamp_in_secs
    milliseconds = (seconds - int(seconds)) * 1000

    return int(minutes), int(seconds), '{:.3f}'.format(milliseconds), '{:02d}:{:02d}:{:.3f}'.format(int(minutes),
                                                                                                    int(seconds),
                                                                                                    milliseconds)

class EchoViewVideoDataset(Data.Dataset):
    """
    This dataset provides short clips (as 2D + t tensors) and their corresponding label describing the view.
    This dataset would normally be useful for classification tasks.

    Arguments
        root (string)   -   the folder where there input files are. The system will expect two files to be here,
                            video_list.txt and annotation_list.txt as described below.

        video_list_file - text file with the names of videos, with a path relative to the root folder. One file per line.

        annotation_list_file - text file with the names of json annotation files, in the same order as the video_list,
                              with a path relative to the root folder. One file per line.

        transform (torch.Transform) - a transform, e.g. for data augmentation, normalization, etc (Default = None)
    """

    def __init__(self, root, video_list_file, annotation_list_file, transform=None):

        self.root = root # folder where the input images are
        self.transform = transform
        self.video_list_file =video_list_file
        self.annotation_list_file = annotation_list_file

        # read the input files to have a list with all the original videos and annotation files
        videoList = os.path.join(root, self.video_list_file)
        annotationList = os.path.join(root, self.annotation_list_file)

        self.video_filenames = [self.root + os.sep+line.strip() for line in open(videoList)]
        self.annotation_filenames = [self.root + os.sep + line.strip() for line in open(annotationList)]

    def __len__(self):
        return len(self.video_filenames)

    def __getitem__(self, index):
        """

        Arguments
            index (int) index position to return the data

        Returns
           Returns  a clip (tensor) with the  4ch view, for file 'index',
        """


        video_name = self.video_filenames[index]
        cap = cv.VideoCapture(video_name)
        if cap.isOpened() == False:
            print('[ERROR] [EchoViewVideoDataset.__getitem__()] Unable to read video ' + video_name)
            exit(-1)

        jsonfile_name = self.annotation_filenames[index]

        start_label_timestamps_ms = []
        end_label_timestamps_ms = []

        with open(jsonfile_name, "r") as json_file:
            json_data = json.load(json_file)
            for key in json_data['metadata']:
                timestamps_of_labels = json_data['metadata'][key]['z']
                # print(timestamps_of_labels)
                start_label_ms = timestamps_of_labels[0] * S2MS
                end_label_ms = timestamps_of_labels[1] * S2MS

                start_label_timestamps_ms.append(start_label_ms)
                end_label_timestamps_ms.append(end_label_ms)

        number_of_labelled_clips = int(len(start_label_timestamps_ms))

        id_clip_to_extract = 0 # TODO: if more than one clip, change this

        # extract the frames of the clip
        frames_torch = []
        cap.set(cv.CAP_PROP_POS_MSEC,start_label_timestamps_ms[id_clip_to_extract])
        while True:
            success, frame = cap.read()
            # in pytorch, channels go first, then height, width
            frame_channelsfirst = np.moveaxis(frame, -1, 0)
            frame_torch = torch.from_numpy(frame_channelsfirst)
            msec = cap.get(cv.CAP_PROP_POS_MSEC)
            if msec > end_label_timestamps_ms[id_clip_to_extract]:
                break
            if not success:
                print('[ERROR] [EchoViewVideoDataset.__getitem__()] Unable to extract frame at ms {} from video '.format(msec))
                break
            frames_torch.append(frame_torch)

        # make a  tensor of the clip

        video_data = torch.stack(frames_torch)

        if self.transform is not None:
                video_data = self.transform(video_data)

        return video_data

