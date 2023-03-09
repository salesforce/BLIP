
import argparse
import os
from os import path
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Union, Tuple, List
from tqdm import tqdm
from data import create_dataset, create_sampler, create_loader

from models.combiner import CombinerModel
from models.blip_fiq import blip_fiq

def generate_randomized_fiq_caption(flattened_captions: List[str]) -> List[str]:
    captions = []
    for i in range(0, len(flattened_captions), 2):
        random_num = random.random()
        if random_num < 0.25:
            captions.append(
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}")
        elif 0.25 < random_num < 0.5:
            captions.append(
                f"{flattened_captions[i + 1].strip('.?, ').capitalize()} and {flattened_captions[i].strip('.?, ')}")
        elif 0.5 < random_num < 0.75:
            captions.append(f"{flattened_captions[i].strip('.?, ').capitalize()}")
        else:
            captions.append(f"{flattened_captions[i + 1].strip('.?, ').capitalize()}")
    return captions

def training_fiq(num_epochs: int, v_dim: int, l_dim: int, dim: int, num_heads: int, clip_model_name: str,
                          combiner_lr: float, clip_bs: int, device, config, **kwargs):
    """
    Train on FashionIQ dataset
    """

    blip_model = blip_fiq(clip_model_name).to(device)

    # Define the train datasets and extract the validation index features
    train_dataset, val_dataset, test_dataset = create_dataset('fashioniq', config)
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers = [None, None, None],
                                batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                num_workers=[4, 4, 4],
                                is_trains=[True, False, False],
                                collate_fns=[None, None, None])


    # Define the combiner
    combiner = CombinerModel(v_dim, l_dim, dim, num_heads).to(device)

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.Adam(combiner.parameters(), lr=combiner_lr)
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        if torch.cuda.is_available():  # RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'

            combiner.train()
            train_bar = tqdm(train_loader, ncols=150)
            for idx, (reference_images, target_images, captions) in enumerate(train_bar):  # Load a batch of triplets
                images_in_batch = reference_images.size(0)

                optimizer.zero_grad()

                reference_images = reference_images.to(device, non_blocking=True)
                target_images = target_images.to(device, non_blocking=True)

                # Randomize the training caption in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
                flattened_captions: list = np.array(captions).T.flatten().tolist()
                text_inputs = generate_randomized_fiq_caption(flattened_captions)

                # Extract the features with CLIP
                with torch.no_grad():
                    # TODO:
                    print("Loss:", blip_model.forward(reference_images, text_inputs, target_images))

def main(args, config):
    device = torch.device(args.device)
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    training_fiq(num_epochs = 1, 
                 v_dim = 1024, 
                 l_dim = 768,
                 dim = 768,
                 num_heads = 12,
                 clip_model_name = '', 
                 combiner_lr = 0.001, 
                 clip_bs = 32, 
                 device = device, 
                 config = config)

    #### Dataset ####
    # print("Creating fashioniq dataset")
    # train_dataset, val_dataset, test_dataset = create_dataset('fashioniq', config)
    # print("Done creating fashioniq dataset")

    # print("Loading fashioniq dataset")
    # samplers = [None, None, None]
    # train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
    #                                                       batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
    #                                                       num_workers=[4, 4, 4],
    #                                                       is_trains=[True, False, False],
    #                                                       collate_fns=[None, None, None])
    # print("Done loading fashioniq dataset")

    # train loader
    # print('train loader')
    # for reference_image, target_image, image_captions in train_loader:
    #     print(reference_image.size())
    #     print(target_image.size())
    #     print(image_captions)
    #     break

    # # val loader
    # print('val loader')
    # for reference_name, target_name, image_captions in val_loader:
    #     print(reference_name)
    #     print(target_name)
    #     print(image_captions)
    #     break

    # # test loader
    # print('test loader')
    # for reference_name, reference_image, image_captions in test_loader:
    #     print(reference_name)
    #     print(reference_image.size())
    #     print(image_captions)
    #     break

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/fashioniq.yaml')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    main(args, config)