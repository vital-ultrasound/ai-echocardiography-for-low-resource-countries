import argparse

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def msec_to_timestamp(current_timestamp: float):
    """
    Convert millisecond variable to a timestamp variable with the format minutes, seconds and milliseconds
    """
    minutes = int(current_timestamp / 1000 / 60)
    seconds = int(np.floor(current_timestamp / 1000) % 60)
    ms = current_timestamp - np.floor(current_timestamp / 1000) * 1000
    current_contour_frame_time = '{:02d}:{:02d}:{:.3f}'.format(minutes, seconds, ms)
    return current_contour_frame_time

def maks_for_captured_us_image(image_frame_array_3ch: np.ndarray):
    """
    Mask pixels outside of scanning sector
    """

    image_frame_array_BW_1ch = cv.cvtColor(image_frame_array_3ch, cv.COLOR_RGB2GRAY)
    mask = np.zeros_like(image_frame_array_BW_1ch)
    top_square_for_review_number_mask = np.array([(275, 5), (296, 5), (296, 25), (275, 25)])
    scan_arc_mask = np.array([(1050, 120), (1054, 120), (1800, 980), (300, 980)])
    cv.fillPoly(mask, [top_square_for_review_number_mask,scan_arc_mask], (255, 255, 0))
    maskedImage = cv.bitwise_and(image_frame_array_3ch, image_frame_array_3ch, mask=mask)

    return maskedImage



def Video_to_ImageFrame(videofile_in: str, image_frames_path: str, bounds=None):
    """
     Computes Channel Measurements per Frame
     bounds: (start_x  ,start_y, width, heigh )
    """
    cap = cv.VideoCapture(videofile_in)
    if cap.isOpened() == False:
        print('Unable to read video ' + videofile_in)

    # Get parameters of input video
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = (cap.get(cv.CAP_PROP_FPS))
    nframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    # Print video features
    print(f'  ')
    print(f'  Frame_height={frame_height},  frame_width={frame_width} fps={fps} nframes={nframes} ' )
    print(f'  ')

    image_frame_index = 0
    rg, rb, gb = [], [], []
    nnz_rg, nnz_rb, nnz_gb = [], [], []
    nz_rg, nz_rb, nz_gb = [], [], []

    while True:
        success, image_frame_array_3ch_i = cap.read()
        if not success:
            break
        frame_msec = cap.get(cv.CAP_PROP_POS_MSEC)
        frame_timestamp = msec_to_timestamp(frame_msec)

        if image_frame_index%2000 == 0:
            print(f'  Frame_index/number_of_frames={image_frame_index}/{nframes},  frame_timestamp={frame_timestamp}')
            print(image_frames_path + '/nframes{:05d}.png'.format(image_frame_index))

            # cropped_image_frame = image_frame[int(bounds[1]):int(bounds[1] + bounds[3]), int(bounds[0]):int(bounds[0] + bounds[2]), :]
            masked_image_frame_array_3ch_i = maks_for_captured_us_image(image_frame_array_3ch_i)

            Rch_image_frame_array = masked_image_frame_array_3ch_i[..., 2].astype(float)
            Gch_image_frame_array = masked_image_frame_array_3ch_i[..., 1].astype(float)
            Bch_image_frame_array = masked_image_frame_array_3ch_i[..., 0].astype(float)


            # image wide rgb
            rg.append(np.mean(np.abs(Rch_image_frame_array - Gch_image_frame_array)))
            rb.append(np.mean(np.abs(Rch_image_frame_array - Bch_image_frame_array)))
            gb.append(np.mean(np.abs(Gch_image_frame_array - Bch_image_frame_array)))
            # n pixels not gray
            nnz_rg.append(np.count_nonzero(np.abs(Rch_image_frame_array - Gch_image_frame_array) > 1))
            nnz_rb.append(np.count_nonzero(np.abs(Rch_image_frame_array - Bch_image_frame_array) > 1))
            nnz_gb.append(np.count_nonzero(np.abs(Gch_image_frame_array - Bch_image_frame_array) > 1))
            # statistics of non gray pixels
            nz_rg.append(np.mean((Rch_image_frame_array - Gch_image_frame_array)[np.abs(Rch_image_frame_array - Gch_image_frame_array) > 1]))
            nz_rb.append(np.mean((Rch_image_frame_array - Bch_image_frame_array)[np.abs(Rch_image_frame_array - Bch_image_frame_array) > 1]))
            nz_gb.append(np.mean((Gch_image_frame_array - Bch_image_frame_array)[np.abs(Gch_image_frame_array - Bch_image_frame_array) > 1]))

            font = cv.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (0, 0, 255)
            thickness = 2
            masked_image_frame_array_3ch_i = cv.putText(masked_image_frame_array_3ch_i, 'R-G: {:.2f}'.format(rg[-1]), (50, 100), font, fontScale, color, thickness,
                               cv.LINE_AA)
            masked_image_frame_array_3ch_i = cv.putText(masked_image_frame_array_3ch_i, 'R-B: {:.2f}'.format(rb[-1]), (50, 150), font, fontScale, color, thickness,
                               cv.LINE_AA)
            masked_image_frame_array_3ch_i = cv.putText(masked_image_frame_array_3ch_i, 'G-B: {:.2f}'.format(gb[-1]), (50, 200), font, fontScale, color, thickness,
                               cv.LINE_AA)
            masked_image_frame_array_3ch_i = cv.putText(masked_image_frame_array_3ch_i, 'R-G nnz: {}'.format(nnz_rg[-1]), (50, 250), font, fontScale, color, thickness,
                               cv.LINE_AA)
            masked_image_frame_array_3ch_i = cv.putText(masked_image_frame_array_3ch_i, 'R-B nnz: {}'.format(nnz_rb[-1]), (50, 300), font, fontScale, color, thickness,
                               cv.LINE_AA)
            masked_image_frame_array_3ch_i = cv.putText(masked_image_frame_array_3ch_i, 'G-B nnz: {}'.format(nnz_gb[-1]), (50, 350), font, fontScale, color, thickness,
                               cv.LINE_AA)
            masked_image_frame_array_3ch_i = cv.putText(masked_image_frame_array_3ch_i, 'R-G nz: {:.2f}'.format(nz_rg[-1]), (50, 400), font, fontScale, color, thickness,
                               cv.LINE_AA)
            masked_image_frame_array_3ch_i = cv.putText(masked_image_frame_array_3ch_i, 'R-B nz: {:.2f}'.format(nz_rb[-1]), (50, 450), font, fontScale, color, thickness,
                               cv.LINE_AA)
            masked_image_frame_array_3ch_i = cv.putText(masked_image_frame_array_3ch_i, 'G-B nz: {:.2f}'.format(nz_gb[-1]), (50, 500), font, fontScale, color, thickness,
                               cv.LINE_AA)

            cv.imwrite(image_frames_path + '/nframes{:05d}.png'.format(image_frame_index),
                       masked_image_frame_array_3ch_i)

        image_frame_index += 1

    plt.subplot(1, 3, 1)
    plt.plot(rg, 'r-', label='r-g')
    plt.plot(rb, 'g-', label='r-b')
    plt.plot(gb, 'b-', label='g-b')
    plt.legend()
    plt.title('Average Absolute difference')
    plt.subplot(1, 3, 2)
    plt.plot(nnz_rg, 'r-', label='r-g')
    plt.plot(nnz_rb, 'g-', label='r-b')
    plt.plot(nnz_gb, 'b-', label='g-b')
    plt.title('NNZ in subtraction')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(nz_rg, 'r-', label='r-g')
    plt.plot(nz_rb, 'g-', label='r-b')
    plt.plot(nz_gb, 'b-', label='g-b')
    plt.title('Average difference over NZNNZ')
    plt.legend()
    plt.savefig(image_frames_path + '/output.jpg')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videofile_in', required=True, help='Specify videofile_in')
    parser.add_argument('--image_frames_path', required=True, help='Specify image_frames_path')
    parser.add_argument('--bounds', required=False, help='Specify bounds', nargs='+', type=int)
    args = parser.parse_args()

    Video_to_ImageFrame(args.videofile_in, args.image_frames_path, args.bounds)

## BLURS
# # # print(image.shape)
#     plt.imshow(masked_image_frame_array_3ch_i)
#     plt.show()


