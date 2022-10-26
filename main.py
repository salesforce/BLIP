from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from models.blip import blip_decoder
from fastapi import FastAPI


device = 'cuda:0'
app = FastAPI()


def load_image_from_url(img_url, image_size, device):
	raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')   

	w,h = raw_image.size
	# display(raw_image.resize((w//5,h//5)))

	transform = transforms.Compose([
		transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
		transforms.ToTensor(),
		transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
		])
	image = transform(raw_image).unsqueeze(0).to(device)
	return image


@app.get('/image_captioning/{img_url}')
async def exec_image_captioning(img_url: str):
	try:
		image_size = 384
		image = load_image_from_url(img_url, image_size, device)

		model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
			
		model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
		model.eval()
		model = model.to(device)

		with torch.no_grad():
			# beam search
			caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
			# nucleus sampling
			# caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
			return {'caption': caption[0]}
	except Exception as e:
		return {'Error': e}
