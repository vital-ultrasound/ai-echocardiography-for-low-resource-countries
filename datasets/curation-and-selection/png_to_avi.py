import cv2
import os

# image_folder = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T1/cropped_us_image/clip001"
# videofileName = "clip001.avi"
# video_name =   "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T1/cropped_us_image/animations/"

# image_folder = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T2/cropped_us_image/clip001"
# videofileName = "clip001.avi"
# video_name =   "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T2/cropped_us_image/animations/"

# image_folder = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T2/cropped_us_image/clip002"
# videofileName = "clip001.avi"
# video_name = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T2/cropped_us_image/animations/"

# image_folder = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T2/cropped_us_image/clip003"
# videofileName = "clip001.avi"
# video_name = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T2/cropped_us_image/animations/"

# image_folder = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T3/cropped_us_image/clip001"
# videofileName = "clip001.avi"
# video_name =   "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T3/cropped_us_image/animations/"

image_folder = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T3/cropped_us_image/clip002"
videofileName = "clip002.avi"
video_name =   "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T3/cropped_us_image/animations/"


if not os.path.isdir(os.path.join(video_name, videofileName)):
    try:
        os.makedirs(video_name)
    except FileExistsError:
        pass

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
fps = 5
video = cv2.VideoWriter(video_name+videofileName, fourcc, fps, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()