import argparse

import yaml
from torch.utils.data import DataLoader

from source.dataloaders.EchocardiographicVideoDataset import EchoVideoDataset, ViewVideoDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Specify config.yml')
    args = parser.parse_args()

    with open(args.config, 'r') as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)

    dataset = ViewVideoDataset(config['participants_videos_path'])
    video_index = 79 #79: /01NVb-003-072/T3/01NVb-003-072-3-echo.mp4
    data = dataset[video_index]

    print(f' {type(data)}, {data.size()} ')

    # my_dloader = DataLoader(data,
    #                     batch_size=200,
    #                     shuffle=True,
    #                     num_workers=4,
    #                     pin_memory=True
    #                     )
    #
    # for (idx, batch) in enumerate(my_dloader):
    #     print(f' Index: {idx}')
    #     print(f' {type(batch)}, {batch.size()} ')
    #     print(f' Batch: {batch}')
    #
