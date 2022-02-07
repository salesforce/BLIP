## BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

<img src="BLIP.gif" width="700">

This is the PyTorch code of the <a href="https://arxiv.org/abs/2201.12086">BLIP paper</a>. The code has been tested on PyTorch 1.10.
To install the dependencies, run <pre/>pip install -r requirements.txt</pre> 

Catalog:
- [x] Inference demo
- [x] Pre-trained and finetuned checkpoints
- [x] Finetuning code for Image-Text Retrieval, Image Captioning, VQA, and NLVR2
- [x] Pre-training code
- [x] Download of bootstrapped pre-training datasets 


### Inference demo:
Run our interactive demo using <a href="https://colab.research.google.com/github/salesforce/BLIP/blob/main/demo.ipynb">Colab notebook</a> (no GPU needed).
The demo includes code for: (1) image captioning, (2) open-ended visual question answering, (3) multimodal / unimodal feature extraction.

Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/BLIP)

### Pre-trained checkpoints:
Num. pre-train images | BLIP w/ ViT-B | BLIP w/ ViT-B and CapFilt-L | BLIP w/ ViT-L 
--- | :---: | :---: | :---: 
14M | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth">Download</a>| - | -
129M | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth">Download</a>| <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base.pth">Download</a> | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth">Download</a>

### Finetuned checkpoints:
Task | BLIP w/ ViT-B | BLIP w/ ViT-B and CapFilt-L | BLIP w/ ViT-L 
--- | :---: | :---: | :---:
Image-Text Retrieval (COCO) | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth">Download</a>| - | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth">Download</a>
Image-Text Retrieval (Flickr30k) | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_flickr.pth">Download</a>|  - | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_flickr.pth">Download</a>
Image Captioning (COCO) | - | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth">Download</a>| <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth">Download</a> | 
VQA | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_vqa.pth">Download</a>| <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_vqa.pth">Download</a> | - 
NLVR2 | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_nlvr.pth">Download</a>| - | - 


To log all your metrics to a [**Weights and Biases Dashboard**](https://wandb.ai/) [![Explore-in W&B](https://img.shields.io/badge/Explore--in-W%26B-%23FFBE00)](https://wandb.ai/sauravm/BLIP) set the `--use_wandb` flag to True while running the scripts and pass in your API key when prompted. To further customize the process you can set the `--wandb_project` ("BLIP" by default) and `--wandb_entity` flags.

### Image-Text Retrieval:
1. Download COCO and Flickr30k datasets from the original websites, and set 'image_root' in configs/retrieval_{dataset}.yaml accordingly.
2. To evaluate the finetuned BLIP model on COCO, run:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py \
--config ./configs/retrieval_coco.yaml \
--output_dir output/retrieval_coco \
--evaluate</pre> 
3. To finetune the pre-trained checkpoint using 8 A100 GPUs, first set 'pretrained' in configs/retrieval_coco.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth". Then run:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py \
--config ./configs/retrieval_coco.yaml \
--output_dir output/retrieval_coco </pre> 

### Image-Text Captioning:
1. Download COCO and NoCaps datasets from the original websites, and set 'image_root' in configs/caption_coco.yaml and configs/nocaps.yaml accordingly.
2. To evaluate the finetuned BLIP model on COCO, run:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_caption.py --evaluate</pre> 
3. To evaluate the finetuned BLIP model on NoCaps, generate results with: (evaluation needs to be performed on official server)
<pre>python -m torch.distributed.run --nproc_per_node=8 eval_nocaps.py </pre> 
4. To finetune the pre-trained checkpoint using 8 A100 GPUs, first set 'pretrained' in configs/caption_coco.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base.pth". Then run:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_caption.py </pre> 

### VQA:
1. Download VQA v2 dataset and Visual Genome dataset from the original websites, and set 'vqa_root' and 'vg_root' in configs/vqa.yaml.
2. To evaluate the finetuned BLIP model, generate results with: (evaluation needs to be performed on official server)
<pre>python -m torch.distributed.run --nproc_per_node=8 train_vqa.py --evaluate</pre> 
3. To finetune the pre-trained checkpoint using 16 A100 GPUs, first set 'pretrained' in configs/vqa.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base.pth". Then run:
<pre>python -m torch.distributed.run --nproc_per_node=16 train_vqa.py </pre> 

### NLVR2:
1. Download NLVR2 dataset from the original websites, and set 'image_root' in configs/nlvr.yaml.
2. To evaluate the finetuned BLIP model, run
<pre>python -m torch.distributed.run --nproc_per_node=8 train_nlvr.py --evaluate</pre> 
3. To finetune the pre-trained checkpoint using 16 A100 GPUs, first set 'pretrained' in configs/nlvr.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth". Then run:
<pre>python -m torch.distributed.run --nproc_per_node=16 train_nlvr.py </pre> 

### Finetune with ViT-L:
In order to finetune a model with ViT-L, simply change the config file to set 'vit' as large. Batch size and learning rate may also need to be adjusted accordingly (please see the paper's appendix for hyper-parameter details). <a href="https://github.com/facebookresearch/fairscale">Gradient checkpoint</a> can also be activated in the config file to reduce GPU memory usage. 

### Pre-train:
1. Prepare training json files where each json file contains a list. Each item in the list is a dictonary with two key-value pairs: {'image': path_of_image, 'caption': text_of_image}. 
2. In configs/pretrain.yaml, set 'train_file' as the paths for the json files .
3. Pre-train the model using 8 A100 GPUs:
<pre>python -m torch.distributed.run --nproc_per_node=8 pretrain.py --config ./configs/Pretrain.yaml --output_dir output/Pretrain </pre> 

### Pre-training datasets download:
We provide bootstrapped pre-training datasets as json files. Each json file contains a list. Each item in the list is a dictonary with two key-value pairs: {'url': url_of_image, 'caption': text_of_image}. 

Image source | Filtered web caption | Filtered synthetic caption | Filtered synthetic caption by ViT-L
--- | :---: | :---: | :---:
CC3M+CC12M+SBU |  <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/ccs_filtered.json">Download</a>|  <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/ccs_synthetic_filtered.json">Download</a>|  <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/ccs_synthetic_filtered_large.json">Download</a>
LAION115M | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/laion_filtered.json">Download</a>|  <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/laion_synthetic_filtered.json">Download</a>|  <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/laion_synthetic_filtered_large.json">Download</a>

### Citation
If you find this code to be useful for your research, please consider citing.
<pre>
@misc{li2022blip,
      title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation}, 
      author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
      year={2022},
      eprint={2201.12086},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}</pre>

### Acknowledgement
The implementation of BLIP relies on resources from <a href="https://github.com/salesforce/ALBEF">ALBEF</a>, <a href="https://github.com/huggingface/transformers">Huggingface Transformers</a>, and <a href="https://github.com/rwightman/pytorch-image-models/tree/master/timm">timm</a>. We thank the original authors for their open-sourcing.
