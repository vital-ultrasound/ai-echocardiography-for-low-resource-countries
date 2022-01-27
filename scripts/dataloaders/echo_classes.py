import argparse
import torch
import yaml

from source.dataloaders.EchocardiographicVideoDataset import EchoClassesDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Specify config.yml')
    args = parser.parse_args()

    with open(args.config, 'r') as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = EchoClassesDataset(main_data_path=config['main_data_path'],
                                 participant_videos_list=config['participant_videos_list'],
                                 participant_path_json_list=config['participant_path_json_list'],
                                 crop_bounds_for_us_image=config['crop_bounds_for_us_image'],
                                 clip_duration=config['n_frames'],
                                 device=device,
                                 max_background_duration_in_secs = 10
                                 )

    ## USAGE
    print(f'Number of clips: {len(dataset)}')
    print(f'Load two clips: ')
    clip_index_a = 0  # this must be within the dataset length
    clip_index_b = 17  # this must be within the dataset length
    data = dataset[clip_index_a]
    data = dataset[clip_index_b]
