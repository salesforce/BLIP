
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from data import create_dataset, create_sampler, create_loader

def main(args, config):
    device = torch.device(args.device)
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating fashioniq dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('fashioniq', config)
    print("Done creating fashioniq dataset")

    print("Loading fashioniq dataset")
    samplers = [None, None, None]
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4, 4, 4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])
    print("Done loading fashioniq dataset")

    # train loader
    print('train loader')
    for out in train_loader:
        print(len(out['caption_1']))
        print(out['caption_1'])
        print(out['caption_2']) 
        print(out['target_img'])
        print(out['candidate_img'])
        break

    # val loader
    print('val loader')
    for out in val_loader:
        print(len(out['caption_1']))
        print(out['caption_1'])
        print(out['caption_2']) 
        print(out['target_img'])
        print(out['candidate_img'])
        break

    # test loader
    print('test loader')
    for out in test_loader:
        print(len(out['caption_1']))
        print(out['caption_1'])
        print(out['caption_2']) 
        print(out['candidate_img'])
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/fashioniq.yaml')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    main(args, config)

