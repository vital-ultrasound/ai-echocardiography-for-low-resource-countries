import argparse
import os

import cv2
import yaml
from tqdm import tqdm


def conver_pngframes_to_avi(image_folder: str,
                            videofileName: str,
                            video_name: str,
                            fps: int,
                            ) -> None:
    """
    Convert image frames in png format to an avi file with a given fps.
    """
    if not os.path.isdir(os.path.join(video_name, videofileName)):
        try:
            os.makedirs(video_name)
        except FileExistsError:
            pass

    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video = cv2.VideoWriter(video_name + videofileName, fourcc, fps, (width, height))

    for image in tqdm(images):
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Specify config.yml with paths files')
    args = parser.parse_args()

    with open(args.config, 'r') as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)

    conver_pngframes_to_avi(config['image_folder'], config['videofileName'], config['video_name'], config['fps'])
