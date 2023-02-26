## BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

## Announcement: BLIP is now officially integrated into [LAVIS](https://github.com/salesforce/LAVIS) - a one-stop library for language-and-vision research and applications!

<img src="BLIP.gif" width="700">

This is the PyTorch code of the <a href="https://arxiv.org/abs/2201.12086">BLIP paper</a> [[blog](https://blog.salesforceairesearch.com/blip-bootstrapping-language-image-pretraining/)]. The code has been tested on PyTorch 1.10.
To install the dependencies, run <pre/>pip install -r requirements.txt</pre>

Catalog:
- [x] Inference demo
- [x] Pre-trained and finetuned checkpoints
- [x] Finetuning code for Image-Text Retrieval, Image Captioning, VQA, and NLVR2
- [x] Pre-training code
- [x] Zero-shot video-text retrieval
- [x] Download of bootstrapped pre-training datasets


### Inference demo:
Run our interactive demo using [Colab notebook](https://colab.research.google.com/github/salesforce/BLIP/blob/main/demo.ipynb) (no GPU needed).
The demo includes code for:
1. Image captioning
2. Open-ended visual question answering
3. Multimodal / unimodal feature extraction
4. Image-text matching

Try out the [Web demo](https://huggingface.co/spaces/Salesforce/BLIP), integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio).

Replicate web demo and Docker image is also available at [![Replicate](https://replicate.com/salesforce/blip/badge)](https://replicate.com/salesforce/blip)

### Pre-trained checkpoints:
Num. pre-train images | BLIP w/ ViT-B | BLIP w/ ViT-B and CapFilt-L | BLIP w/ ViT-L
--- | :---: | :---: | :---:
14M | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth">Download</a>| - | -
129M | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth">Download</a>| <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth">Download</a> | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth">Download</a>

### Finetuned checkpoints:
Task | BLIP w/ ViT-B | BLIP w/ ViT-B and CapFilt-L | BLIP w/ ViT-L
--- | :---: | :---: | :---:
Image-Text Retrieval (COCO) | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth">Download</a>| - | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth">Download</a>
Image-Text Retrieval (Flickr30k) | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_flickr.pth">Download</a>|  - | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_flickr.pth">Download</a>
Image Captioning (COCO) | - | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth">Download</a>| <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth">Download</a> |
VQA | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_vqa.pth">Download</a>| <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth">Download</a> | -
NLVR2 | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_nlvr.pth">Download</a>| - | -


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
4. To finetune the pre-trained checkpoint using 8 A100 GPUs, first set 'pretrained' in configs/caption_coco.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth". Then run:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_caption.py </pre>

### VQA:
1. Download VQA v2 dataset and Visual Genome dataset from the original websites, and set 'vqa_root' and 'vg_root' in configs/vqa.yaml.
2. To evaluate the finetuned BLIP model, generate results with: (evaluation needs to be performed on official server)
<pre>python -m torch.distributed.run --nproc_per_node=8 train_vqa.py --evaluate</pre>
3. To finetune the pre-trained checkpoint using 16 A100 GPUs, first set 'pretrained' in configs/vqa.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth". Then run:
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

### Zero-shot video-text retrieval:
1. Download MSRVTT dataset following the instructions from https://github.com/salesforce/ALPRO, and set 'video_root' accordingly in configs/retrieval_msrvtt.yaml.
2. Install [decord](https://github.com/dmlc/decord) with <pre>pip install decord</pre>
3. To perform zero-shot evaluation, run
<pre>python -m torch.distributed.run --nproc_per_node=8 eval_retrieval_video.py</pre>

### Pre-training datasets download:
We provide bootstrapped pre-training datasets as json files. Each json file contains a list. Each item in the list is a dictonary with two key-value pairs: {'url': url_of_image, 'caption': text_of_image}.

Image source | Filtered web caption | Filtered synthetic caption by ViT-B | Filtered synthetic caption by ViT-L
--- | :---: | :---: | :---:
CC3M+CC12M+SBU |  <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/ccs_filtered.json">Download</a>|  <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/ccs_synthetic_filtered.json">Download</a>|  <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/ccs_synthetic_filtered_large.json">Download</a>
LAION115M | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/laion_filtered.json">Download</a>|  <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/laion_synthetic_filtered.json">Download</a>|  <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/laion_synthetic_filtered_large.json">Download</a>

### Citation
If you find this code to be useful for your research, please consider citing.
<pre>
@inproceedings{li2022blip,
      title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
      author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
      year={2022},
      booktitle={ICML},
}</pre>

### Acknowledgement
The implementation of BLIP relies on resources from <a href="https://github.com/salesforce/ALBEF">ALBEF</a>, <a href="https://github.com/huggingface/transformers">Huggingface Transformers</a>, and <a href="https://github.com/rwightman/pytorch-image-models/tree/master/timm">timm</a>. We thank the original authors for their open-sourcing.

## Data preparation


```
project_base_path

└───  cirr_dataset
       └─── train
            └─── 0
                | train-10108-0-img0.png
                | train-10108-0-img1.png
                | train-10108-1-img0.png
                | ...

            └─── 1
                | train-10056-0-img0.png
                | train-10056-0-img1.png
                | train-10056-1-img0.png
                | ...

            ...

       └─── dev
            | dev-0-0-img0.png
            | dev-0-0-img1.png
            | dev-0-1-img0.png
            | ...

       └─── test1
            | test1-0-0-img0.png
            | test1-0-0-img1.png
            | test1-0-1-img0.png
            | ...

       └─── cirr
            └─── captions
                | cap.rc2.test1.json
                | cap.rc2.train.json
                | cap.rc2.val.json

            └─── image_splits
                | split.rc2.test1.json
                | split.rc2.train.json
                | split.rc2.val.json


```

### Preparing CIRR

1. Clone CIRR dataset `git clone -b cirr_dataset https://github.com/Cuberick-Orion/CIRR.git cirr`
2. Raw images from `https://lil.nlp.cornell.edu/resources/NLVR2/`

### Preparing Fashion200k

The instructions to download Fashion 200k can be found [here](https://github.com/xthan/fashion-200k)

For the generated test_queries.txt, we use the one generated by the authors of [TIRG paper](https://github.com/google/tirg) which can be found [here](https://storage.googleapis.com/image_retrieval_css/test_queries.txt). 

The fashion-200k directory should follow the following structure and have these files.

```
/fashion-200k/labels/*.txt
/fashion-200k/women/<category>/<caption>/<id>/*.jpeg
/fashion-200k/test_queries.txt`
```
