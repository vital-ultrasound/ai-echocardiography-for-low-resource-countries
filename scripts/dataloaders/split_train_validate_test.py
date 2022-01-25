import argparse

from source.helpers.various import split_train_validate_sets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_echodataset_path', required=True, help='Specify video_echodataset_path')
    parser.add_argument('--ntraining', required=True, help='Factor of training', type=float)
    args = parser.parse_args()
    split_train_validate_sets(args.video_echodataset_path, args.ntraining)

    # USAGE
    # python split_train_validate_test.py --video_echodataset_path /home/mx19/datasets/vital-us/echocardiography/videos-echo/ --ntraining 0.8
