'''
This script is created to be run under DEBUG mode, to inspect the model pipeline & intermediate variables
Most functions are copied from demo.ipynb
'''

from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# function to load a image as example
def load_demo_image(image_size, device):
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    w, h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

##% -----------------------------------------------------------------------------
# Extract image & text features with pretrained encoders
from models.blip import blip_feature_extractor

image_size = 224
image = load_demo_image(image_size=image_size, device=device)

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth'

model = blip_feature_extractor(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

caption = 'a woman sitting on the beach with a dog'

multimodal_feature = model(image, caption, mode='multimodal')[0, 0]
image_feature = model(image, caption, mode='image')[0, 0]
text_feature = model(image, caption, mode='text')[0, 0]
##% -----------------------------------------------------------------------------
# Tell whether a image & a sentence are matched or not (binary classification)
from models.blip_itm import blip_itm

image_size = 384
image = load_demo_image(image_size=image_size, device='cpu')

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'

model = blip_itm(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device='cpu')

caption = 'a woman sitting on the beach with a dog'

print('text: %s' % caption)

itm_output = model(image, caption, match_head='itm')
itm_score = torch.nn.functional.softmax(itm_output, dim=1)[:, 1]
print('The image and text is matched with a probability of %.4f' % itm_score)

itc_score = model(image, caption, match_head='itc')
print('The image feature and text feature has a cosine similarity of %.4f' % itc_score)
##% -----------------------------------------------------------------------------
# Decode image caption from an image
from models.blip import blip_decoder

image_size = 384
image = load_demo_image(image_size=image_size, device=device)

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'

model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

with torch.no_grad():
    # beam search
    caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
    # nucleus sampling
    # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
    print('caption: ' + caption[0])