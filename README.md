# ML-Danbooru: Anime image tags detector



## [Modified] Usage:



1. clone repo & install deps:

```bash
git clone https://github.com/troph-team/ml-danbooru-tagger && cd ml-danbooru-tagger
pip install -r requirements.txt
```



2. get a model:

```bash
wget https://huggingface.co/kiriyamaX/mld-caformer/resolve/main/ml_caformer_m36_dec-5-97527.ckpt
```



3. start tagging:
	-  tags will be saved as json file next to the tagged folder
	-   for more args available see the python file

```bash
python demo_ca.py--ckpt ml_caformer_m36_dec-5-97527.ckpt --thr 0.65 --image_size 448  --bs 12 --data {OUR_PATH}
```



full commands:
```bash
(base) root@DSK11N3:~/dev/ml-danbooru-tagger# python demo_ca.py --help
usage: demo_ca.py [-h] [--data DATA] [--ckpt CKPT] [--class_map CLASS_MAP] [--model_name MODEL_NAME]
                  [--num_classes NUM_CLASSES] [--image_size N] [--thr N] [--keep_ratio KEEP_RATIO] [--bs BS]
                  [--use_ml_decoder USE_ML_DECODER] [--fp16] [--ema] [--frelu FRELU] [--xformers XFORMERS]
                  [--decoder_embedding DECODER_EMBEDDING] [--num_layers_decoder NUM_LAYERS_DECODER]
                  [--num_head_decoder NUM_HEAD_DECODER] [--num_queries NUM_QUERIES]
                  [--scale_skip SCALE_SKIP] [--out_type OUT_TYPE]

ML-Danbooru Demo

options:
  -h, --help            show this help message and exit
  --data DATA
  --ckpt CKPT
  --class_map CLASS_MAP
  --model_name MODEL_NAME
  --num_classes NUM_CLASSES
  --image_size N        input image size
  --thr N               threshold value
  --keep_ratio KEEP_RATIO
  --bs BS
  --use_ml_decoder USE_ML_DECODER
  --fp16
  --ema
  --frelu FRELU
  --xformers XFORMERS
  --decoder_embedding DECODER_EMBEDDING
  --num_layers_decoder NUM_LAYERS_DECODER
  --num_head_decoder NUM_HEAD_DECODER
  --num_queries NUM_QUERIES
  --scale_skip SCALE_SKIP
  --out_type OUT_TYPE
```









## [original readme below] Introduction

An anime image tag detector based on modified [ML-Decoder](https://github.com/Alibaba-MIIL/ML_Decoder).
Model trained with cleaned [danbooru2021](https://gwern.net/danbooru2021).

+ Designed a new TResNet-D structure as backbone to enhance the learning of low-level features.
+ Replace the ReLU in backbone with [FReLU](https://arxiv.org/pdf/2007.11824.pdf).
+ Using learnable queries for transformer decoder.

## Model Structure

![](./imgs/ml_danbooru.png)

## Model-Zoo
https://huggingface.co/7eu7d7/ML-Danbooru

## Usage
Download the model and run below command:
```bash
python demo.py --data <path to image or directory> --model_name tresnet_d --num_of_groups 32 --ckpt <path to ckpt> --thr 0.7 --image_size 640 
```

Keep the image ratio invariant:
```bash
python demo.py --data <path to image or directory> --model_name tresnet_d --num_of_groups 32 --ckpt <path to ckpt> --thr 0.7 --image_size 640 --keep_ratio True
```

### ML_CAFormer
```bash
python demo_ca.py --data <path to image or directory> --model_name caformer_m36 --ckpt <path to ckpt> --thr 0.7 --image_size 448
```