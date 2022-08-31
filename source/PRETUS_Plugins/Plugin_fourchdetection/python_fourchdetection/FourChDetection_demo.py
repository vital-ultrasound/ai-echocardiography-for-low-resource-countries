
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import FourChDetection_worker as worker
import SimpleITK as sitk
import time
import skimage
from skimage import transform
import cv2


import cv2 as cv

def crop2(frames, crop_bounds):
    croped = frames[:, crop_bounds[1]:crop_bounds[1] + crop_bounds[3], crop_bounds[0]:crop_bounds[0] + crop_bounds[2]]  # [:, 25:414, 96:584]
    return croped

def resize(frames, size=(64, 64)):
    frames_out = []
    for i in range(frames.shape[0]):
        #frames_out.append(cv2.resize(frames[i, :, :], size, interpolation=Image.ANTIALIAS))
        frames_out.append(skimage.transform.resize(frames[i, :, :], output_shape=size, anti_aliasing=False, preserve_range=True))
    frames_out = np.array(frames_out)

    return frames_out

def normalize(frames, correct_studio_swing=False):
    frames = frames.astype(np.float)
    if correct_studio_swing:
        #np.save('/home/ag09/data/press.npy', frames)
        frames[frames < 3] = 3
        #np.save('/home/ag09/data/postss.npy', frames)

    for i in range(frames.shape[0]):
        # Min Max normalization
        _min = np.amin(frames[i, :, :])
        frames[i, :, :] = frames[i, :, :] - _min
        _max = np.amax(frames[i, :, :]) + 1e-6
        frames[i, :, :] = frames[i, :, :] / _max

    #np.save('/home/ag09/data/postn.npy', frames)
    #exit(-1)
    return frames

def resample(image, desired_size):
    size = desired_size
    origin = image.GetOrigin()
    spacing = [(s2 - 1) * sp2 / (s1 - 1) for s1, s2, sp2 in zip(desired_size, image.GetSize(), image.GetSpacing())]

    ref = sitk.Image(size, sitk.sitkInt8)
    ref.SetOrigin(origin)
    ref.SetSpacing(spacing)

    # resample
    identity_transform = sitk.AffineTransform(image.GetDimension())
    identity_transform.SetIdentity()
    image = sitk.Resample(image, ref, identity_transform, sitk.sitkLinear, 0)

    return image


def read_data_from_video(video_file, start_frame, duration, crop_bounds, desired_size):
    cap = cv.VideoCapture(video_file)
    frames_np = []
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(duration):
        success, frame = cap.read()
        if not success:
            print('[ERROR] [EchoViewVideoDataset.__getitem__()] Unable to extract frame  from video')
            break
        # in pytorch, channels go first, then height, width
        frame_channelsfirst = np.moveaxis(frame, -1, 0)
        frames_np.append(frame_channelsfirst)

    # make a  tensor of the clip
    video_data = np.stack(frames_np)
    # convert to gray

    video_datag = video_data[:,0,...]
    # remove text and labels
    aa = (video_data[:, 0, ...].astype(np.float64) - video_data[:, 2, ...].astype(np.float64)).squeeze() ** 2
    # plt.subplot(1,3,1)
    # plt.imshow(aa[-1,...])
    # plt.subplot(1, 3, 2)
    # plt.imshow(video_datag[-1, ...])
    video_datag[aa > 0] = 0
    video_datag[...,:60,:150] = 0

    # plt.subplot(1, 3, 3)
    # plt.imshow(video_datag[-1, ...])
    # plt.show()

    video_datag[ aa >0 ]=0



    #cv.imwrite('{}_original.png'.format(video_file[:-4]), video_data[0,...])
    #video_data = crop(video_data)
    video_datag = crop2(video_datag, crop_bounds)
    #cv.imwrite('{}_cropped.png'.format(video_file[:-4]), video_data[0, ...])
    video_datag = resize(video_datag, desired_size)
    #cv.imwrite('{}_cropped_resized.png'.format(video_file[:-4]), video_data[0, ...])
    #video_data = normalize(video_data)
    #cv.imwrite('{}_cropped_resized_normalised.png'.format(video_file[:-4]), video_data[0, ...] * 255)

    return video_datag

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--InputVideoID', required=True, help='Specify USB ID', type=int)
    args = parser.parse_args()

    #print(f'Input video: {args.InputVideoID}')
    crop_bounds_ = (96, 25, 488, 389)

    start_frame_ = 20 #?
    #duration_ = 10 #?
    BATCH_SIZE_OF_CLIPS = 10
    NUMBER_OF_FRAMES_PER_SEGMENT_IN_A_CLIP = 5;
    desired_size_ = (128, 128)
    frames = read_data_from_video(video_file= args.InputVideoID, start_frame= start_frame_, duration= NUMBER_OF_FRAMES_PER_SEGMENT_IN_A_CLIP, crop_bounds = crop_bounds_, desired_size= desired_size_)
    print(f' FourCHDetection_demo:main() Acquired frames.shape {frames.shape}')

    print_model_arquitecture_and_name = True # False # True
    modelfolder = '../python_fourchdetection/models'
    modelfilename_ = 'basicVGG2D_04layers_model_BATCH_SIZE_OF_CLIPS_10.pth'
    worker.initialize(input_size= desired_size_, model_path= modelfolder, modelname = modelfilename_, verb=print_model_arquitecture_and_name)

    print(f'========================')
    print(f'Run the model inference')
    startt = time.time_ns() #Use time_ns() to avoid the precision loss caused by the float type.
    #predictions, aw, attentions = worker.dowork(frames) #predictions?, aw?, attentions?
    predictions = worker.dowork(frames)
    print(f' Predictions {predictions}')
    endt = time.time_ns()
    print(f'inference elapsed time: {(endt - startt)/1000000}ms')
    print(f'========================')


    #
    # ##\/ TOREVIEW
    # # # find peaks
    # # awm0 = (attentions[0, 0,...] * (frames[-1,...]>0) ).astype(np.uint8)
    # # (_, _, _, maxLoc0) = cv2.minMaxLoc(awm0)
    # # awm1 = (attentions[0,-1, ...] * (frames[-1, ...] > 0)).astype(np.uint8)
    # # (_, _, _, maxLoc1) = cv2.minMaxLoc(awm1)
    # #
    # # arrowlength = 15
    # #
    # # dir = np.array(maxLoc1)-np.array(maxLoc0)
    # # dir = dir/np.linalg.norm(dir)
    # # p1 = np.array(maxLoc1)- dir * 1
    # # p0= np.array(maxLoc1) - dir * (arrowlength +1)
    # # fr = frames[-1,...]
    # # fr2 = frames[-1,...]
    # # cv2.arrowedLine(fr2, tuple(p0.astype(np.int)), tuple(p1.astype(np.int)), (255, 255, 255), 1, tipLength=0.3)
    # ## plot attention
    # # plt.figure()
    # # plt.subplot(1, 2, 1)
    # # plt.imshow(frames, cmap='gray')
    # # plt.imshow(aw, alpha=aw.astype(np.float) / 255, cmap='jet')
    # # plt.subplot(1, 2, 2)
    # # plt.imshow(fr2)
    # # plt.show()
    # # #pred_segmentation = pred_segmentation[0, ...].squeeze()
    # # #pred_segmentation_t = (pred_segmentation.cpu().numpy() > 0.5).astype(np.uint8)
    # # #pred_area = np.sum(pred_segmentation_t)
    # ##/\ TOREVIEW

if __name__ == '__main__':
    main()

