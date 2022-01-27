import argparse
import torch
import yaml
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from source.dataloaders.EchocardiographicVideoDataset import EchoClassesDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Specify config.yml')
    args = parser.parse_args()

    with open(args.config, 'r') as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)

    # define some static transforms, i.e. transforms that apply to the entire data with
    # no change. These transforms are not augmentation.
    pretransform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=config['im_size']),
        transforms.ToTensor(),  # this normalizes in
    ]
    )

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    dataset = EchoClassesDataset(main_data_path=config['main_data_path'],
                                 participant_videos_list=config['participant_videos_list'],
                                 participant_path_json_list=config['participant_path_json_list'],
                                 crop_bounds_for_us_image=config['crop_bounds_for_us_image'],
                                 clip_duration_nframes=config['n_frames'],
                                 device=device,
                                 max_background_duration_in_secs=config['max_background_duration_in_secs'],
                                 pretransform=pretransform,
                                 )

    # create a dataloader that will serve the batches over the epochs
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=True)

    ## USAGE
    print(f'Number of clips: {len(dataset)}')
    print(f'Load two clips: ')
    clip_index_a = 0  # this must be within the dataset length
    clip_index_b = 15  # this must be within the dataset length
    data_a = dataset[clip_index_a]
    data_b = dataset[clip_index_b]

    print('Display the two clips:')

    labelnames = ('bck', '4Ch')

    plt.figure()
    for f in range(data_a[0].shape[1]):
        plt.subplot(2, data_a[0].shape[1], f+1)
        plt.imshow(data_a[0][0, f, ...], cmap='gray')
        plt.title('{} {}'.format(labelnames[data_a[1]], f))
    for f in range(data_b[0].shape[1]):
        plt.subplot(2, data_b[0].shape[1], f+data_b[0].shape[1]+1)
        plt.imshow(data_b[0][0, f, ...], cmap='gray')
        plt.title('{} {}'.format(labelnames[data_b[1]], f))
    plt.show()

    # -----------------------------------------
    # Do a loop as if we were training a model
    for epoch in range(config['max_epochs']):
        for step, data in enumerate(dataloader):
            clip = data[0]
            label = data[1].to(device)
        print('Done for epoch {}'.format(epoch))