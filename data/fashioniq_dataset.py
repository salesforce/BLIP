import json
from PIL import Image
from os import mkdir, path, listdir
from torch.utils.data import Dataset
import gdown
import zipfile
from typing import List

class FashionIQDataset(Dataset):
    def __init__(self, data_path, split: str, dress_types: List[str], mode: str, transform: callable):
        self.data_path = data_path
        self.split = split
        self.dress_types = dress_types
        self.mode = mode
        self.transform = transform

        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        # get the images from the web
        self.download_images(data_path)
        
        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets = []
        for dress_type in dress_types:
            with open(path.join(data_path, 'captions', f'cap.{dress_type}.{split}.json')) as f:
                self.triplets.extend(json.load(f))

         # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(path.join(data_path, 'image_splits', f'split.{dress_type}.{split}.json')) as f:
                self.image_names.extend(json.load(f))
        
        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized, found {len(self.triplets)} data.")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        try:
            if self.mode == 'relative':
                image_captions = self.triplets[idx]['captions']
                reference_name = self.triplets[idx]['candidate']

                if self.split == 'train':
                    reference_image_path = path.join(self.data_path, 'images', f"{reference_name}.png")
                    reference_image = self.load_image(reference_image_path)

                    target_name = self.triplets[idx]['target']
                    target_image_path = path.join(self.data_path, 'images', f"{target_name}.png")
                    target_image = self.load_image(target_image_path)
                    
                    return reference_image, target_image, image_captions

                elif self.split == 'val':
                    target_name = self.triplets[idx]['target']
                    return reference_name, target_name, image_captions

                elif self.split == 'test':
                    reference_image_path = path.join(self.data_path, 'images', f"{reference_name}.png")
                    reference_image = self.load_image(reference_image_path)
                    return reference_name, reference_image, image_captions

            elif self.mode == 'classic':
                image_name = self.image_names[idx]
                image_path = self.data_path / 'images' / f"{image_name}.png"
                image = self.transform(Image.open(image_path))
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")
    
    def load_image(self, file_path):
        img = Image.open(file_path)
        img = img.convert('RGB')
        img = self.transform(img)
        return img
    
    def download_images(self, data_path):
        if not path.exists(data_path):
            mkdir(data_path)
            file_id="1IvXYP_j5iT2QVyowpw-AgI6q2xQ4pHMy"
            url = f'https://drive.google.com/uc?id={file_id}'
            output = path.join(data_path, 'fashionIQ_dataset.zip')
            gdown.download(url, output, quiet=False)
            with zipfile.ZipFile(path.join(data_path, "fashionIQ_dataset.zip"),"r") as zip_ref:
                zip_ref.extractall(data_path)
