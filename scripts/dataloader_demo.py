import cv2 as cv
import numpy as np

import datasets.dataloaders.EchocardiographicVideoDataset as EchoDatasets

dataset = EchoDatasets.EchoViewVideoDataset(root='/home/ag09/data/VITAL/echo', video_list_file='video_list.txt',
                                            annotation_list_file='annotation_list.txt')

# the dataset will give us a the clip at index 0.
# The dataset class will take care of looking into the json file for the start and end
# and return only the 4chamber part.
sample_index = 0
data = dataset[sample_index]

# Now to check that this is what we wanted, let's convert it to a video clip. There is a lot of things still to do.
# For example, the dataloader maybe needs to send also some details about the original video (like framerate),
# Also need to incorporate crops, etc.

# TODO: change this by a local folder of yours!
output_filename = '/home/ag09/data/VITAL/echo/output_video.avi'
print('Write to {}'.format(output_filename))
# load your frames
fps = 30
frame_size = data.shape[-1:-3:-1]
fourcc = cv.VideoWriter_fourcc(*'MJPG')
writer = cv.VideoWriter(output_filename, fourcc, fps, tuple(frame_size))
# and write your frames in a loop if you want
for i in range(data.shape[0]):
    frame = data[i, ...].numpy()
    frame_channels_last = np.moveaxis(frame, 0, -1)
    writer.write(frame_channels_last)
writer.release()

print('Done!')
