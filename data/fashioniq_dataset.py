import json
from PIL import Image
from os import mkdir, path, listdir
from torch.utils.data import Dataset
import gdown
import zipfile

class FashionIQDataset(Dataset):
    def __init__(self, data_path, split="train", transform=None):
        super(FashionIQDataset, self).__init__()
        
        self.split = split        
        self.img_root = path.join(data_path, 'images')
        self.cap_root = path.join(data_path, 'captions')
        self.transform = transform

        if split not in ['train', 'val', 'test']:
            return Exception("Invalid split parameter, use train, val or test")
        
        # get the images from the web
        self.download_images(data_path)
        
        # get the captions from the json
        print("Loading captions")
        self.imgs = []
        for file in listdir(self.cap_root):
            if split in file:
                f = open(path.join(self.cap_root, file))
                caption_list = json.load(f)
                for e in caption_list:
                    if self.split != 'test':
                        img = {
                            'target_file' : path.join(self.img_root, e['target'] + '.png'),
                            'candidate_file' : path.join(self.img_root, e['candidate'] + '.png'),
                            'captions': e['captions']
                        }
                    else:
                        img = {
                            'candidate_file' : path.join(self.img_root, e['candidate'] + '.png'),
                            'captions': e['captions']
                        }
                    self.imgs.append(img)
        print(f'FashionIQ {split}:', len(self.imgs), 'images')

    def get_all_texts(self):
        texts = []
        for img in self.imgs:
            for c in img['captions']:
                texts.append(c)
            return texts
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        out = {}
        out['caption_1'] = self.imgs[idx]['captions'][0]
        out['caption_2'] = self.imgs[idx]['captions'][1]
        if self.split != 'test': 
            out['target_img'] = self.load_image(self.imgs[idx]['target_file'])
        out['candidate_img'] = self.load_image(self.imgs[idx]['candidate_file'])
        return out
    
    def load_image(self, file_path):
        img = Image.open(file_path)
        img = img.convert('RGB')
        if self.transform:
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

if __name__ == '__main__':
    data_path = 'fashion_iq'
    fashion_iq = FashionIQDataset(data_path, 'test')