import os
import argparse
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms

from ml_danbooru_tagger.data.path_dataset import PathDataset
from ml_danbooru_tagger.helper_functions.helper_functions import crop_fix
from ml_danbooru_tagger.models import create_model
from tqdm.auto import tqdm

from huggingface_hub import hf_hub_download

import json
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def make_args():
    parser = argparse.ArgumentParser(description='ML-Danbooru Demo')
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--class_map', type=str, default='./class.json')
    parser.add_argument('--model_name', default='caformer_m36')
    parser.add_argument('--num_classes', default=12547)
    parser.add_argument('--image_size', default=448, type=int,
                        metavar='N', help='input image size')
    parser.add_argument('--thr', default=0.6, type=float,
                        metavar='N', help='threshold value for probability to be saved in json dict')
    parser.add_argument('--keep_ratio', type=str2bool, default=False)
    parser.add_argument('--bs', type=int, default=16)

    parser.add_argument('--str_thr', default=0.7, type=float,
                    metavar='N', help='threshold value for probability to be saved in string')

    # ML-Decoder
    parser.add_argument('--use_ml_decoder', default=0, type=int)
    parser.add_argument('--fp16', action="store_true", default=False)
    parser.add_argument('--ema', action="store_true", default=False)

    parser.add_argument('--frelu', type=str2bool, default=True)
    parser.add_argument('--xformers', type=str2bool, default=False)

    # CAFormer
    parser.add_argument('--decoder_embedding', default=384, type=int)
    parser.add_argument('--num_layers_decoder', default=4, type=int)
    parser.add_argument('--num_head_decoder', default=8, type=int)
    parser.add_argument('--num_queries', default=80, type=int)
    parser.add_argument('--scale_skip', default=1, type=int)

    parser.add_argument('--out_type', type=str, default='json')

    args = parser.parse_args()
    return args

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
from logging import getLogger

class Demo:
    def __init__(self, args):

        self.logger = getLogger(__name__)
        self.args=args

        self.logger.info('creating model {}...'.format(args.model_name))
        args.model_path = None
        model = create_model(args, load_head=True).to(device)
        if args.ckpt:
            state = torch.load(args.ckpt, map_location='cpu')
        else:
            ckpt_path = self.download_model()
            state = torch.load(ckpt_path, map_location='cpu')
        
        if args.ema:
            state = state['ema']
        elif 'model' in state:
            state=state['model']
        model.load_state_dict(state, strict=True)

        self.model = model.to(device).eval()
        #######################################################
        self.logger.info('model load done')

        if args.keep_ratio:
            self.trans = transforms.Compose([
                transforms.Resize(args.image_size),
                crop_fix,
                transforms.ToTensor(),
            ])
        else:
            self.trans = transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                ])

        self.load_class_map()

    
    def download_model(self):
        REPO_ID = "kiriyamaX/mld-caformer"
        CKPT_FILE = "ml_caformer_m36_dec-5-97527.ckpt"
        self.logger.info(f"Loading model file from {REPO_ID}")
        model_path = hf_hub_download(repo_id=REPO_ID, filename=CKPT_FILE)
        return model_path
    
    def download_class_map(self):
        REPO_ID = "kiriyamaX/mld-caformer"
        CLASS_FILE = "mld_caformer_mapping_dict.json"
        self.logger.info(f"Loading class map file from {REPO_ID}")
        model_path = hf_hub_download(repo_id=REPO_ID, filename=CLASS_FILE)
        return model_path    
    
    def load_class_map(self):
        if not self.args.class_map or not os.path.exists(self.args.class_map):
            self.logger.info("class map not provided (or not found), downloading from huggingface hub")
            self.args.class_map = self.download_class_map()

        print(f"using class map from {self.args.class_map}")
        with open(self.args.class_map, 'r') as f:
            self.class_map = json.load(f)
            self.logger.info(f"loaded {len(self.class_map)} classes")

    def load_data(self, path):
        img = Image.open(path).convert('RGB')
        img = self.trans(img)
        return img

    def infer_one(self, img):
        with torch.cuda.amp.autocast(enabled=self.args.fp16):
            img = img.unsqueeze(0)
            output = torch.sigmoid(self.model(img)).cpu().view(-1)
        pred = torch.where(output > self.args.thr)[0].numpy()

        cls_list = [(self.class_map[str(i)], float(output[i])) for i in pred]
        return cls_list


    def _get_single_entry_dict(self, img_root, cls_list, config={}):
        cls_list.sort(reverse=True, key=lambda x: x[1])        
        tag_probs = {name.replace('_', ' '): round(prob, 8) for name, prob in cls_list}
        tag_str = ', '.join([name.replace('_', ' ') for name, prob in cls_list if prob >= self.args.str_thr])

        entry = {
            "tag_str": tag_str,
            "tag_probs": tag_probs,
            "config": config,
        }
        return entry
    
    
    @torch.no_grad()
    def infer(self, path):

        # Assuming --data is the root directory
        root_dir = self.args.data  

        if os.path.isfile(path):  # single image
            img = self.load_data(path).to(device)
            cls_list = self.infer_one(img)
            curr_entry = self._get_single_entry_dict(root_dir, cls_list)
            return curr_entry
        
        else: # directory
            tag_dict = {}
            img_list = [os.path.join(path, x) for x in os.listdir(path) if x[x.rfind('.'):].lower() in IMAGE_EXTENSIONS]
            for item in tqdm(img_list):
                img = self.load_data(item).to(device)
                cls_list = self.infer_one(img)
                curr_entry = self._get_single_entry_dict(root_dir, cls_list)

                # add to tag_dict (with relative path as key)
                relative_path = os.path.relpath(item, root_dir)
                tag_dict[relative_path] = curr_entry

            if self.args.out_type == 'json':
                json_path = os.path.join(os.path.dirname(path), os.path.basename(path) + '_mld.json')
                self.logger.info(f"writing json to {json_path}")
                with open(json_path, 'w', encoding='utf8') as f:
                    f.write(json.dumps(tag_dict, indent=2, ensure_ascii=False))
                
                self.logger.info(f"json written to {json_path}")
                
            return tag_dict



    @torch.no_grad()
    def infer_batch(self, path, bs=8):
        root_dir = self.args.data  # Assuming --data is the root directory
        tag_dict = {}
        img_list = [os.path.join(path, x) for x in os.listdir(path) if x[x.rfind('.'):].lower() in IMAGE_EXTENSIONS]
        dataset = PathDataset(img_list, self.trans)
        loader = torch.utils.data.DataLoader(dataset, batch_size=bs, num_workers=4, shuffle=False)

        for imgs, path_list in tqdm(loader):
            imgs = imgs.to(device)

            with torch.cuda.amp.autocast(enabled=self.args.fp16):
                output_batch = torch.sigmoid(self.model(imgs)).cpu()

            for output, img_path in zip(output_batch, path_list):
                pred = torch.where(output > self.args.thr)[0].numpy()
                cls_list = [(self.class_map[str(i)], float(output[i])) for i in pred]
                cls_list.sort(reverse=True, key=lambda x: x[1])

                tag_probs = {name.replace('_', ' '): round(prob, 8) for name, prob in cls_list}
                tag_str = ', '.join([name.replace('_', ' ') for name, prob in cls_list if prob >= self.args.str_thr])

                entry = {
                    "tag_str": tag_str,
                    "tag_probs": tag_probs,
                    "config": {}
                }

                relative_path = os.path.relpath(img_path, root_dir)
                tag_dict[relative_path] = entry

        if self.args.out_type == 'json':            
            json_path = os.path.join(os.path.dirname(path), os.path.basename(path) + '_mld.json')
            with open(json_path, 'w', encoding='utf8') as f:
                f.write(json.dumps(tag_dict, indent=2, ensure_ascii=False))
        
        return tag_dict


class Tagger:
    def __init__(self, model_name='caformer_m36', ckpt='', image_size=448, thr=0.6, bs=16):
        args = argparse.Namespace(
            data='',
            ckpt=ckpt,
            class_map='./class.json',
            model_name=model_name,
            num_classes=12547,
            image_size=image_size,
            thr=thr,
            keep_ratio=False,
            bs=bs,
            str_thr=0.7,
            use_ml_decoder=0,
            fp16=False,
            ema=False,
            frelu=True,
            xformers=False,
            decoder_embedding=384,
            num_layers_decoder=4,
            num_head_decoder=8,
            num_queries=80,
            scale_skip=1,
            out_type='json'
        )
        self.demo = Demo(args)

    def tag_image(self, image_path):
        self.demo.infer(image_path)

    def tag_images_batch(self, images_dir):
        self.demo.infer_batch(images_dir, self.demo.args.bs)


import time
def mld_cli(img_dir):
    """
    simplistic version of ml-danbooru tagger cli to be used in other projects
    :param img_dir:
    :return:
    """
    start = time.time()
    tagger = Tagger(bs=16)
    tagger.tag_images_batch(img_dir)
    print(f"[ml-danbooru] DONE; Time taken: {time.time() - start:.4f}s")



import argparse

def make_args_safe():
    """
    Create an argparse.Namespace with default values without parsing command-line arguments.
    This avoids conflicts in environments like Jupyter notebooks where unintended command-line arguments may be present.
    """
    args = argparse.Namespace(
        data='',
        ckpt='',
        class_map='',
        model_name='caformer_m36',
        num_classes=12547,
        image_size=448,
        thr=0.6,
        keep_ratio=False,
        bs=16,
        str_thr=0.7,
        use_ml_decoder=0,
        fp16=False,
        ema=False,
        frelu=True,
        xformers=False,
        decoder_embedding=384,
        num_layers_decoder=4,
        num_head_decoder=8,
        num_queries=80,
        scale_skip=1,
        out_type='json'
    )
    return args


def infer_batch_with_defaults(image_path, class_map_path='', batch_size = None):
    """
    Function to run batch inference using default arguments with specified image path, bypassing command-line argument parsing.
    Args:
    - image_path (str): The path to the directory containing images to be processed in batch.
    - class_map_path (str): The path to the class map JSON file. If not provided, will download from Hugging Face Hub.
    """
    # Use the safe version of make_args to get default arguments
    default_args = make_args_safe()

    # Update the 'data' and 'class_map' argument
    default_args.data = image_path
    if class_map_path:
        default_args.class_map = class_map_path
    
    if batch_size:
        default_args.bs = batch_size

    # Create the Demo instance with updated arguments
    demo = Demo(default_args)
    tag_dict = demo.infer_batch(default_args.data, default_args.bs)
    return tag_dict




#python demo_ca.py --data imgs/t1.jpg --model_name caformer_m36 --ckpt ckpt/ml_caformer_m36_dec-5-97527.ckpt --thr 0.7 --image_size 448
if __name__ == '__main__':
    args = make_args()
    demo = Demo(args)
    if args.bs>1:
        cls_list = demo.infer_batch(args.data, args.bs)
    else:
        cls_list = demo.infer(args.data)

    if cls_list is not None:
        cls_list.sort(reverse=True, key=lambda x: x[1])
        print(', '.join([f'{name}:{prob:.3}' for name, prob in cls_list]))
        print(', '.join([name for name, prob in cls_list]))