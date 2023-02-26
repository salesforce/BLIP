
import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import cv2
import clip

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
    print("Creating fashion200k dataset")
    train_dataset, test_dataset = create_dataset('fashion200k', config)

    samplers = [None, None]
    train_loader, test_loader = create_loader([train_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']],
                                                          num_workers=[4,4],
                                                          is_trains=[True, False],
                                                          collate_fns=[None, None])

    # train loader
    print('train loader')
    for out in train_loader:
        #print(out['source_img_id'])
        print(out['source_img_data'].size()) 
        print(out['source_caption'])
        #print(out['target_img_id'] )
        print(out['target_img_data'].size())
        print(out['target_caption'] )
        print(out['mod'])
        break


    # test loader
    print('test loader')
    for out in test_loader:
        #print(out['source_img_id'])
        print(out['source_img_data'].size()) 
        print(out['source_caption'])
        #print(out['target_img_id'] )
        print(out['target_img_data'].size())
        print(out['target_caption'] )
        print(out['mod'])
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/fashion200k.yaml')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)


    main(args, config)

