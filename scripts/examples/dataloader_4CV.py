import argparse
import yaml
from source.dataloaders.EchocardiographicVideoDataset import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Specify config.yml')
    args = parser.parse_args()

    with open(args.config, 'r') as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)

    dataset = EchoViewVideoDataset(config['participant_videos_path'], config['participant_path_json_files'])
    video_index = 2
    data = dataset[video_index]
