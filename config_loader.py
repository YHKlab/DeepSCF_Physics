import argparse
import os

from omegaconf import OmegaConf

def get_args():

    source_directory = os.path.dirname(os.path.abspath(__file__))

    # input yaml files
    parser = argparse.ArgumentParser()
    parser.add_argument('--default', type=str,
                        default=f'{source_directory}/config/input.yaml')
    parser.add_argument('--input', type=str, default='./input.yaml')
    args = parser.parse_args()

    # default config
    if os.path.exists(args.default):
        default_config = OmegaConf.load(args.default)
    else:
        raise FileNotFoundError(f'Cannot find default input file: {args.default}')

    # user config
    if os.path.exists(args.input):
        user_config = OmegaConf.load(args.input)
        input_config = OmegaConf.merge(default_config, user_config)
    else:
        print(f'Input file not found: {args.input}')
        print('Using the default configuration only.')
        input_config = default_config

    return input_config
