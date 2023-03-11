
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
from models.blip_pretrain import BLIP_Pretrain

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

 # init combiner
    combiner = CombinerModel(
        config['v_dim'], config['l_dim'], config['dim'], config['num_heads']).to(device)

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.Adam(combiner.parameters(), lr=config['combiner_lr'])
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # init BLIP pretrained model to use their encoders
    print('Loading pretrained BLIP')
    blip = BLIP_Pretrain().to(device)
    print('BLIP loaded succesfuly')

    print('========== Start training loop ========== ')

    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch}:")

        if torch.cuda.is_available():
            combiner.train()
            train_bar = tqdm(train_loader, ncols=150)

            for idx, (reference_images, target_images, image_captions) in enumerate(train_bar):
                
                optimizer.zero_grad()

                reference_images = reference_images.to(device, non_blocking=True)
                target_images = target_images.to(device, non_blocking=True)
                text_inputs = blip.tokenizer(image_captions, padding='max_length', truncation=True, max_length=config['max_length'], return_tensors="pt").to(device)

                # Extract the features with BLIP here
                with torch.no_grad():

                    reference_image_features = blip.visual_encoder(reference_images)
                    target_image_features = blip.visual_encoder(target_images)

                    text_features = blip.text_encoder(text_inputs.input_ids, attention_mask=text_inputs.attention_mask,
                                                      return_dict=True, mode='text')

                    print()
                    print("----------------Result-------------------")
                    print(reference_image_features.size())
                    print(target_image_features.size())
                    print(text_features['last_hidden_state'].size())
                    

                with torch.cuda.amp.autocast():

                    combiner_out_v, combiner_out_l = combiner(reference_image_features,
                                                              text_features['last_hidden_state'], text_inputs.attention_mask[0, :])

                    v_proj = nn.Linear(config['v_dim'], config['dim']).to(device)
                    v = v_proj(combiner_out_v)

                    loss = crossentropy_criterion(v, target_image_features)
                    print(loss)
                    break
                
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/fashioniq.yaml')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    main(args, config)