import cv2 as cv
import numpy as np

def VideoToSlidingVideo(videofile_in, videofile_out, boxwidth = (60, 50) ):
    """Crop each frame with a floating bounding box that moves around the image.
    The size of the crop is defined by "2*boxwidth". The crop box "floats" around the
    video in a linear trajectory between two random start and end points"""

    cap = cv.VideoCapture(videofile_in)
    # Check if video opened successfully
    if (cap.isOpened() == False):
        print('Unable to read video ' + videofile_in)

    # get parameters of input video, which will be the same in the video output
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    nframes = cap.get(cv.CAP_PROP_FRAME_COUNT)


    p0 = np.array((int(np.random.uniform(boxwidth[0], frame_width-boxwidth[0])), int(np.random.uniform(boxwidth[1], frame_height-boxwidth[1]))))
    p1 = np.array((int(np.random.uniform(boxwidth[0], frame_width-boxwidth[0])), int(np.random.uniform(boxwidth[1], frame_height-boxwidth[1]))))


    out = cv.VideoWriter(videofile_out, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (2 * boxwidth[0], 2 * boxwidth[1]))

    # checks whether frames were extracted
    success = True
    i=0
    while True:
        # vidObj object calls read
        # function extract frames
        success, image = cap.read()
        if not success:
            break

        # get image region
        p = ((p0*(nframes-i) + p1*i)/nframes).astype(np.int) # centre of the box

        imCrop = image[p[1]-boxwidth[1]:p[1]+boxwidth[1], p[0]-boxwidth[0]:p[0]+boxwidth[0], :]

        # Write the frame into the file 'output.avi'
        out.write(imCrop)
        i +=1

    cap.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # Calling the function
    # Input parameters
    #videofile_in = '/home/ag09/data/VITAL/ucf11/v_biking_01_03.avi'
    #videofile_in = '/home/ag09/data/VITAL/ucf11/v_shooting_02_04.avi'
    videofile_in = '/home/ag09/data/VITAL/ucf11/v_walk_dog_05_05.avi'

    #videofile_out = '/home/ag09/data/VITAL/ucf11/v_biking_01_03_sliding.avi'
    #videofile_out = '/home/ag09/data/VITAL/ucf11/v_shooting_02_04_sliding.avi'
    videofile_out = '/home/ag09/data/VITAL/ucf11/v_walk_dog_05_05_sliding.avi'

    # optional input parameters
    # new video size

    boxwidth = (80, 60)  # rows x columns the size of the video will be twice this
    VideoToSlidingVideo(videofile_in, videofile_out, boxwidth)
