import argparse

from source.helpers.various import split_train_validate_sets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--echodataset_path', required=True, help='Specify echodataset_path')
    parser.add_argument('--data_list_output_path', required=True, help='Specify data_list_output_path')
    parser.add_argument('--ntraining', required=True, help='Factor of training', type=float)
    args = parser.parse_args()
    split_train_validate_sets(args.echodataset_path, args.data_list_output_path, args.ntraining)

    # TERMINAL USAGE:
    # python split_train_validate_test.py --video_echodataset_path $HOME/datasets/vital-us/echocardiography/videos-echo/ --text_label_output_path $HOME/repositories/echocardiography/scripts/config_files/ --ntraining 0.8
