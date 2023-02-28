import json
import random
from PIL import Image
from os import mkdir, path, listdir
import torch
from torch.utils.data import Dataset
from fashioniq_loader import get_images


class FashionIQDataset(Dataset):
    def __init__(self, img_root, cap_root, split="train", transform=None):
        super(FashionIQDataset, self).__init__()

        self.split = split        
        self.img_root = img_root
        self.cap_root = cap_root
        self.transform = transform

        if split not in ['train', 'val', 'test']:
            return Exception("Invalid split parameter, use train, val or test")
        
        # get the images
        get_images()
        print("Loading images")
        self.images = {}
        for file in listdir(img_root):
            img = Image.open(path.join(img_root, file))
            if self.transform:
                img = self.transform(img)
            label = file.replace('.jpg','')
            self.images[label] = img

        # get the captions from the json
        print("Loading captions")
        self.imgs = []
        for file in listdir(cap_root):
            if split in file:
                f = open(path.join(cap_root, file))
                tmp = json.load(f)
                self.imgs += tmp

        # get the images into the dictionary
        for img in self.imgs:
            target_label = img['target']
            if target_label not in self.images: 
                print(f"Image {target_label} not found")
                continue
            img['target'] = self.images[target_label]

            candidate_label = img['candidate']
            if candidate_label not in self.images: 
                print(f"Image {candidate_label} not found")
                continue
            img['candidate'] = self.images[candidate_label]

        print('FashionIQ:', len(self.imgs), 'images')

    def get_all_texts(self):
        texts = []
        for img in self.imgs:
            for c in img['captions']:
                texts.append(c)
            return texts
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]

# cur_path = path.dirname(path.abspath(__file__))
# img_root = path.join(cur_path, 'images')
# cap_root = path.join(cur_path, 'captions')
# fashion_iq = FashionIQDataset(img_root, cap_root)