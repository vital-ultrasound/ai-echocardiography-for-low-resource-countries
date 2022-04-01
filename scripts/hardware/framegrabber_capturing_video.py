import argparse

import cv2


def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


def CaptureVideoTests(frame_width: int, frame_height: int, frames_per_second: int):
    ID_GRABBER = 2

    id = ID_GRABBER
    cap = cv2.VideoCapture(id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'V', '1', '2'))
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', '1', '2'))
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('B', 'G', 'R', '3'))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, frames_per_second)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Check if the device is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open video source {}".format(id))

    # print properties of te capture
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    bs = cap.get(cv2.CAP_PROP_BUFFERSIZE)

    print('fps: {}'.format(fps))
    print('resolution: {}x{}'.format(w, h))  # default 640 x 480
    print('mode: {}'.format(decode_fourcc(fcc)))  # default 640 x 480
    print('Buffer size: {}'.format(bs))  # default 640 x 480

    while (True):
        ret, frame = cap.read()
        if frame is not None:
            cv2.imshow('frame', frame)
        else:
            print('No frame')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_width', required=True, help='Specify width of the image', type=int)
    parser.add_argument('--frame_height', required=True, help='Specify high of the image', type=int)
    parser.add_argument('--frames_per_second', required=True, help='Specify FPS', type=int)
    args = parser.parse_args()

    CaptureVideoTests(args.frame_width, args.frame_height, args.frames_per_second)

if __name__ == '__main__':
    main()
