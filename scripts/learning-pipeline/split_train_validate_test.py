import argparse

import yaml

from source.helpers.various import split_train_validate_sets, concatenating_YAML_via_tags

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Specify config.yml')
    args = parser.parse_args()

    yaml.add_constructor('!join', concatenating_YAML_via_tags)  ## register the tag handler

    with open(args.config, 'r') as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)

    split_train_validate_sets(config['echodataset_path'], config['data_list_output_path'], config['ntraining'], config['randomise_file_list'])
