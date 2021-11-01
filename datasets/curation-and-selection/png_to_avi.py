import cv2
import os

image_folder = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T2/cropped_us_image/clip001"
video_name = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T2/cropped_us_image/animations/clip001.avi"

# image_folder = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T2/cropped_us_image/clip002"
# video_name = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T2/cropped_us_image/animations/clip002.avi"

# image_folder = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T2/cropped_us_image/clip003"
# video_name = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T2/cropped_us_image/animations/clip003.avi"

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
fps = 5
video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()