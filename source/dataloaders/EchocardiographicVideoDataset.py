import json
import os

import cv2 as cv
import numpy as np
import torch
from tqdm import tqdm

from source.helpers.various import timer_func_decorator, msec_to_timestamp, ToImageTensor

# constants
S2MS = 1000


class EchoVideoDataset(torch.utils.data.Dataset):
    """
    ViewVideoDataset Class for Loading Video using torch.utils.data.
    """

    def __init__(
            self,
            participant_videos_path: str,
            participant_path_json_files: str,
            transform=None
    ):
        """
        Arguments

        participant_videos_path (srt):  the folder where there input files are.
        participant_path_json_files (str): text file with the names of json annotation files, in the same order as the video_list.
        transform (torch.Transform): a transform, e.g. for data augmentation, normalization, etc (Default = None)
        """

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

    @timer_func_decorator
    def __getitem__(self, video_index: int):
        """
        Arguments:
            video_index (int): video_index position to return the data

        Returns:
           video_data clip (tensor): vide data clip with the 4ch view, for file 'video_index',
        """

        image_frame_index = 0
        video_name = self.video_filenames[video_index]
        jsonfile_name = self.annotation_filenames[video_index]

        cap = cv.VideoCapture(video_name)
        if cap.isOpened() == False:
            print('[ERROR] [ViewVideoDataset.__getitem__()] Unable to open video ' + video_name)
            exit(-1)

        # Get parameters of input video
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(np.ceil(cap.get(cv.CAP_PROP_FPS)))
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        # Print video features
        print(f'  ')
        print(f'  ')
        print(f'  video_name={video_name}')
        print(f'  Frame_height={frame_height}, frame_width={frame_width} fps={fps} nframes={frame_count} ')
        print(f'  jsonfile_name={jsonfile_name}')

        ## Extracting timestams in json files for labelled of four chamber views (4CV)
        start_label_timestamps_ms = []
        end_label_timestamps_ms = []
        with open(jsonfile_name, "r") as json_file:
            json_data = json.load(json_file)
            if len(json_data['metadata']) == 0:  ## Check if the metadata is empty
                id_clip_to_extract = 0
                print(f'')
                print(f'  No 4CV labels for {json_file_i}')
                print(f'')
                # break
            else:
                for key in json_data['metadata']:
                    timestamps_of_labels = json_data['metadata'][key]['z']
                    start_label_ms = timestamps_of_labels[0] * S2MS
                    end_label_ms = timestamps_of_labels[1] * S2MS
                    start_label_timestamps_ms.append(start_label_ms)
                    end_label_timestamps_ms.append(end_label_ms)

        number_of_labelled_clips = int(len(start_label_timestamps_ms))
        print(f'  number_of_labelled_clips={number_of_labelled_clips}')
        print(f'  ')
        print(f'  ')

        frames_torch = []
        pbar = tqdm(total=frame_count)
        while True:
            success, image_frame_array_3ch_i = cap.read()
            if (success == True):

                frame_msec = cap.get(cv.CAP_PROP_POS_MSEC)
                current_frame_timestamp = msec_to_timestamp(frame_msec)

                frame_torch = ToImageTensor(image_frame_array_3ch_i)

                #### PLAYGROUND
                ## cap.set(cv.CAP_PROP_POS_MSEC, start_label_timestamps_ms[id_clip_to_extract])

                # ## condition for  minute_label
                for clips_i in range(0, number_of_labelled_clips):
                    if (current_frame_timestamp[0] >= int(msec_to_timestamp(start_label_timestamps_ms[clips_i])[0])) & (
                            current_frame_timestamp[0] <= int(msec_to_timestamp(end_label_timestamps_ms[clips_i])[0])):
                        # condition for second label
                        if (current_frame_timestamp[1] >= int(
                                msec_to_timestamp(start_label_timestamps_ms[clips_i])[1])) & (
                                current_frame_timestamp[1] <= int(
                            msec_to_timestamp(end_label_timestamps_ms[clips_i])[1])):
                            # print(
                            #     f'  clip {clips_i}; image_frame_index {image_frame_index}, frame_msec {frame_msec}, current_frame_timestamp {current_frame_timestamp}')
                            frames_torch.append(frame_torch)

                pbar.update(1)
                image_frame_index += 1

            else:
                break

        pbar.close()
        cap.release()

        video_data = torch.stack(frames_torch) # "Fi,C,H,W"
        # video_data = video_data.squeeze() # "Fi,H,W" for one channel

        if self.transform is not None:
            video_data = self.transform(video_data)

        return video_data
