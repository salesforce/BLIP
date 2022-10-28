from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
from PIL import Image
import requests
import torch

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
from models.blip import blip_decoder, blip_vqa, blip_feature_extractor, blip_itm

import uvicorn


app = FastAPI()

device = 'cuda:0'
image_size = 384


def load_image_from_url(img, image_size, device):
	raw_image = Image.open(img).convert('RGB')

	w,h = raw_image.size
	# display(raw_image.resize((w//5,h//5)))

	transform = transforms.Compose([
		transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
		transforms.ToTensor(),
		transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
		])
	image = transform(raw_image).unsqueeze(0).to(device)
	return image


@app.get('/')
def main():
	content = '''
	<body>
	<form action="/upload" enctype="multipart/form-data" method="post">
	<span>Task: </span><input name="task" type="text">
	<br />
	<br />
	<input name="file" type="file">
	<br />
	<br />
	<input type="submit">
	</form>
	</body>
	'''
	return HTMLResponse(content=content)


@app.post('/upload')
async def upload_image(task: str = Form(), file: UploadFile = Form()):
	try:
		img = file.file

		if task == 'image_captioning':
			image = load_image_from_url(img, image_size, device)
			with torch.no_grad():
				# beam search
				caption = image_captioning_model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
				# nucleus sampling
				# caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
				return {'Caption': caption[0]}

		if task == 'vqa':
			image = load_image_from_url(img, image_size, device)
			with torch.no_grad():
				caption = vqa_model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
				return {'Caption': caption[0]}

		if task == 'feature_extraction':
			image = load_image_from_url(img, image_size, device)
			with torch.no_grad():
				caption = feature_extraction_model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
				return {'Caption': caption[0]}

		if task == 'text_matching':
			image = load_image_from_url(img, image_size, device)
			with torch.no_grad():
				caption = image_text_matching_model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
				return {'Caption': caption[0]}

	except Exception as e:
		return {'Error': e}


if __name__ == '__main__':
	image_captioning_model = blip_decoder(pretrained='./models/model_base_capfilt_large.pth', image_size=image_size, vit='base')
	image_captioning_model.eval()
	image_captioning_model = image_captioning_model.to(device)

	vqa_model = blip_vqa(pretrained='./models/model_base_vqa_capfilt_large.pth', image_size=image_size, vit='base')
	vqa_model.eval()
	vqa_model = vqa_model.to(device)

	feature_extraction_model = blip_feature_extractor(pretrained='./models/model_base.pth', image_size=image_size, vit='base')
	feature_extraction_model.eval()
	feature_extraction_model = feature_extraction_model.to(device)

	image_text_matching_model = blip_itm(pretrained='./models/model_base_retrieval_coco.pth', image_size=image_size, vit='base')
	image_text_matching_model.eval()
	image_text_matching_model = image_text_matching_model.to(device)

	uvicorn.run(app, host='0.0.0.0', port=80)
