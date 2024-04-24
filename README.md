# ML-Danbooru: Anime image tags detector
infer_batch_20k_with_defaults.py set up
```
python infer_batch_20k_with_defaults.py --data [data path] --bs 8 --ckpt [ckpt path] --general_thr .75 --characters_thr .85
```
Need to set up ckpt and class_map on s3
##
requirements.txt updated
##
Code to download xformer==0.0.22 with torch 2.1.0 and cu118:
```
pip install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu118
```
Code to download inplace_abn:
```
git clone https://github.com/mapillary/inplace_abn.git
cd inplace_abn
python setup.py install
```
## [Modified] Usage:


**package usage**:

1. install the package: `pip install -r requirements.txt `

2. get labels by importing function from the package:

```python
from ml_danbooru_tagger import infer_batch_with_defaults
bench_tag_dict = infer_batch_with_defaults("bench_images_small")
```


**package usage:**

to build the package:
```bash
python setup.py bdist_wheel
```



**CLI usage:**


1. clone repo & install deps:

```bash
git clone https://github.com/troph-team/ml-danbooru-tagger && cd ml-danbooru-tagger
pip install -r requirements.txt
```



2. start tagging:

   -  automatically downloads models
   -  optimal settings preconfigured
   -  tags will be saved as json file next to the tagged folder

   -   for more args available see the python file

```bash
python demo_ca.py --data {IMAGE_DIR_PATH}
```


modifications:

- better error handling
- can be used as a pacakge on downsteraam projects
- downloads model automatically when not given
- keeps probability of tags along with string repr
- adds batched inference by default
- modified json save path





## [Original Readme]

see: https://github.com/IrisRainbowNeko/ML-Danbooru
