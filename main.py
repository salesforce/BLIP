from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
from PIL import Image
import requests
import torch

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse

from models.blip import blip_decoder, blip_feature_extractor
from models.blip_vqa import blip_vqa
from models.blip_itm import blip_itm

import uvicorn


app = FastAPI()

device = 'cuda:0'
image_size = 384


def load_image(img, image_size, device):
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


@app.get('/')
def main():
	content = '''
	<body>
	<h3>Available options</h3>
	<ul>
	<li>Image Captioning: <i>GET</i> <a href='/image_captioning' target='_blank' style='text-decoration: inherit'> `/img_captioning`</a></li>
	<li>Visual Question Answering: <i>GET</i> <a href='/vqa' target='_blank' style='text-decoration: inherit'>`/vqa`</a></li>
	<li>Feature Extraction: <i>GET</i> <a href='/feature_extraction' target='_blank' style='text-decoration: inherit'> `/feature_extraction`</a></li>
	<li>Image Text Matching: <i>GET</i> <a href='/image_text_matching' target='_blank' style='text-decoration: inherit'> `/image_text_matching`</a></li>
	</ul>
	</body>
	'''
	return HTMLResponse(content=content)


@app.get('/image_captioning')
def main():
	content = '''
	<body>
	<form action="/upload" enctype="multipart/form-data" method="post">
	<input name="task" type="text" value="image_captioning" hidden>
	<input name="file" type="file">
	<br />
	<br />
	<input type="submit">
	</form>
	</body>
	'''
	return HTMLResponse(content=content)


@app.get('/vqa')
def main():
	content = '''
	<body>
	<form action="/upload" enctype="multipart/form-data" method="post">
	<input name="task" type="text" value="vqa" hidden>
	<span>Question: </span><input name="question" type="text">
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


@app.get('/feature_extraction')
def main():
	content = '''
	<body>
	<form action="/upload" enctype="multipart/form-data" method="post">
	<input name="task" type="text" value="feature_extraction" hidden>
	<span>Question: </span><input name="question" type="text">
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


@app.get('/image_text_matching')
def main():
	content = '''
	<body>
	<form action="/upload" enctype="multipart/form-data" method="post">
	<input name="task" type="text" value="text_matching" hidden>
	<span>Question: </span><input name="question" type="text">
	<br />
	<br />
	<span>Mode (<i>itm or itc</i>): </span><input name="mode" type="text" value="itm">
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
async def upload_image(
	task: str = Form(),
	question: [str, None] = Form(),
	caption: [str, None] = Form(),
	mode: [str, None] = Form(),
	match_head: [str, None] = Form(),
	file: UploadFile = Form()
):
	try:
		img = file.file
		image = load_image(img, image_size, device)

		if task == 'image_captioning':
			with torch.no_grad():
				# beam search
				result = image_captioning_model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
				# nucleus sampling
				# caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
				return {'Caption': result[0]}
		if task == 'vqa':
			with torch.no_grad():
				answer = vqa_model(image, question, sample=False, num_beams=3, max_length=20, min_length=5) 
				return {'Answer': answer[0]}
		if task == 'feature_extraction':
			with torch.no_grad():
				result = feature_extraction_model(image, caption, mode) [0, 0]
				return {'Result': result} # ?
		if task == 'text_matching':
			with torch.no_grad():
				if match_head == 'itm':
					itm_output = image_text_matching_model(image, caption, match_head)
					itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
					return {'The image and text is matched with a probability of %.4f'%itm_score}
				elif match_head == 'itc':
					itc_score = image_text_matching_model(image, caption, match_head)
					return {'The image feature and text feature has a cosine similarity of %.4f'%itc_score}

	except Exception as e:
		return {'Error': e}


@app.post('/upload/url')
async def upload_image(task: str, img_url: str):
	try:
		image = load_image_from_url(img_url, image_size, device)

		if task == 'image_captioning':
			with torch.no_grad():
				# beam search
				caption = image_captioning_model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
				# nucleus sampling
				# caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
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
