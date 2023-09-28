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
python demo_ca.py --data {OUR_PATH}
```





modifications:

- downloads model automatically when not given
- keeps probability of tags along with string repr
- adds batched inference by default
- modified json save path





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