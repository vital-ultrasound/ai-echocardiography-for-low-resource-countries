import json
import os

import cv2 as cv
import numpy as np
import torch
import torch.utils.data as Data

# constants
S2MS = 1000

class EchoViewVideoDataset(Data.Dataset):
    """
    This dataset provides clips (as 2D + t tensors) and their corresponding label describing the view.
    This dataset would normally be useful for classification tasks.

    Arguments
        participant_videos_path (srt):  the folder where there input files are.

        participant_path_json_files (str): text file with the names of json annotation files, in the same order as the video_list.

        transform (torch.Transform): a transform, e.g. for data augmentation, normalization, etc (Default = None)
    """

    def __init__(self, participant_videos_path: str, participant_path_json_files: str, transform=None):

        self.participant_videos_path = participant_videos_path
        self.participant_path_json_files = participant_path_json_files
        self.transform = transform

        self.video_filenames = []
        for T_days_i in enumerate(sorted(os.listdir(self.participant_videos_path))):
            days_i_path = self.participant_videos_path + '/' + T_days_i[1]
            for video_file_name_i in sorted(os.listdir(days_i_path)):
                path_video_file_name_i = days_i_path + '/' + video_file_name_i
                if path_video_file_name_i.endswith('.mp4'):
                    self.video_filenames += [path_video_file_name_i]

        self.annotation_filenames = []
        for json_i in enumerate(sorted(os.listdir(self.participant_path_json_files))):
            self.annotation_filenames += [self.participant_path_json_files + '/' + json_i[1]]

    def __len__(self):
        return len(self.video_filenames)

    def __getitem__(self, index: int):
        """

        Arguments:
            index (int): index position to return the data

        Returns
           video_data clip (tensor): vide data clip with the 4ch view, for file 'index',
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

        id_clip_to_extract = 0  # TODO: if more than one clip, change this

        # extract the frames of the clip
        frames_torch = []
        cap.set(cv.CAP_PROP_POS_MSEC, start_label_timestamps_ms[id_clip_to_extract])
        while True:
            success, frame = cap.read()
            # in pytorch, channels go first, then height, width
            frame_channelsfirst = np.moveaxis(frame, -1, 0)
            frame_torch = torch.from_numpy(frame_channelsfirst)
            msec = cap.get(cv.CAP_PROP_POS_MSEC)
            if msec > end_label_timestamps_ms[id_clip_to_extract]:
                break
            if not success:
                print(
                    '[ERROR] [EchoViewVideoDataset.__getitem__()] Unable to extract frame at ms {} from video '.format(
                        msec))
                break
            frames_torch.append(frame_torch)

        # make a  tensor of the clip
        video_data = torch.stack(frames_torch)

        if self.transform is not None:
            video_data = self.transform(video_data)

        return video_data
