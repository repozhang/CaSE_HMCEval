from pathlib import Path
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

from confidnet.utils.logger import get_logger

LOGGER = get_logger(__name__, level="DEBUG")


class AbstractDataLoader:    
    def __init__(self, config_args):
        self.output_folder = config_args['training']['output_folder']
        self.data_dir = config_args['data']['data_dir']
        self.batch_size = config_args['training']['batch_size']
        self.img_size = (config_args['data']['input_size'][0],
                         config_args['data']['input_size'][1],
                         config_args['data']['input_channels'])
        self.augmentations = config_args['training'].get('augmentations', None)
        self.resume_folder = config_args['model']['resume'].parent if isinstance(config_args['model']['resume'], Path) else None
        self.valid_size = config_args['data']['valid_size']
        self.perturbed_folder = config_args['data'].get('perturbed_images', None)
        self.pin_memory = config_args['training']['pin_memory']
        self.num_workers = config_args['training']['num_workers']
        self.train_loader, self.val_loader, self.test_loader = None, None, None


        # Load dataset
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.load_dataset()


    def load_dataset(self):
        pass


