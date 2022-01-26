import argparse

import yaml
from torch.utils.data import DataLoader

from source.dataloaders.EchocardiographicVideoDataset import EchoClassesDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Specify config.yml')
    args = parser.parse_args()

    with open(args.config, 'r') as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)

    dataset = EchoClassesDataset(main_data_path=config['main_data_path'],
                                 participant_videos_list=config['participant_videos_list'],
                                 participant_path_json_list=config['participant_path_json_list'],
                                 crop_bounds=config['crop_bounds'],
                                 clip_duration=config['n_frames'])

    clip_index = 10 # this must be within the dataset length
    data = dataset[clip_index]

    print('data retrieved')
    # print(f' {type(data)}, {data.size()} ')

    # my_dloader = DataLoader(data,
    #                     batch_size=1,
    #                     shuffle=False,
    #                     num_workers=0,
    #                     pin_memory=True
    #                     )

    # for idx_batch, sample_batched in enumerate(my_dloader):
    #     print(f' Index: {idx_batch}')
    #     print(f' {type(sample_batched)}, {sample_batched.size()} ')
    #     # print(f' Batch: {batch}')

