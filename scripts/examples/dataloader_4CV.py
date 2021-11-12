import argparse
import yaml
from datasets.dataloaders.EchocardiographicVideoDataset import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Specify config.yml')
    args = parser.parse_args()

    with open(args.config, 'r') as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)

    dataset = EchoViewVideoDataset(config['participant_datapath'],
                                   config['participant_path_json_files'],
                                   config['video_list_file'],
                                   config['annotation_list_file'])
    sample_index = 0
    data = dataset[sample_index]


