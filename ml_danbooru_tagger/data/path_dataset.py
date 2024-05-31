import torch
from torch.utils.data import Dataset
import boto3
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from torchvision.transforms.functional import pad
from io import BytesIO


class ResizeAndPad:
    def __init__(self, target_size, fill=0, padding_mode='constant'):
        self.target_size = target_size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        # Calculate the scale to resize the image so the long side is self.target_size
        scale = self.target_size / max(img.width, img.height)
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        
        # Resize the image
        img = transforms.Resize((new_height, new_width))(img)
        
        # Calculate padding to make the image square (self.target_size x self.target_size)
        pad_left = (self.target_size - new_width) // 2
        pad_top = (self.target_size - new_height) // 2
        pad_right = self.target_size - new_width - pad_left
        pad_bottom = self.target_size - new_height - pad_top
        
        # Apply padding
        img = pad(img, (pad_left, pad_top, pad_right, pad_bottom), self.fill, self.padding_mode)
        return img


class PathDataset(Dataset):
    def __init__(self, img_list, transform=None):
        self.img_list = img_list
        self.transform = transform
        self.s3 = boto3.client('s3')
        self.IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']  # Supported image formats
        self.valid_indices = self._validate_images()
            
    def _validate_images(self):
        valid_indices = []
        for i, path in enumerate(self.img_list):
            try:
                if path.startswith('s3://'):
                    path = path.replace('s3://', '')
                    bucket = path.split('/')[0]
                    key = '/'.join(path.split('/')[1:])
                    response = self.s3.get_object(Bucket=bucket, Key=key)
                    img = Image.open(BytesIO(response['Body'].read())).convert('RGB')
                    img.load()  # Ensure the image can be fully loaded
                else:
                    with Image.open(path) as img:
                        img.verify()  # Verify the image to check if it's corrupted
                        img = img.convert('RGB')
                        img.load()  # Ensure the image can be fully loaded
                valid_indices.append(i)
            except (IOError, SyntaxError, FileNotFoundError) as e:
                print(f"Skipping corrupted image due to {e.__class__.__name__}: {path}")
            except Exception as e:  # A catch-all for any other unforeseen exceptions
                print(f"Skipping image due to unexpected error ({e.__class__.__name__}): {path}")
        return valid_indices
        
    def __getitem__(self, index):        
        valid_index = self.valid_indices[index]
        path = self.img_list[valid_index]
        
        if path.startswith('s3://'):
            path = path.replace('s3://', '')
            bucket = path.split('/')[0]
            key = '/'.join(path.split('/')[1:])
            response = self.s3.get_object(Bucket=bucket, Key=key)
            img = Image.open(BytesIO(response['Body'].read())).convert('RGB')
        else:
            img = Image.open(path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        return img, path

    def __len__(self):
        return len(self.valid_indices)
# Example usage:
# transform = ResizeAndPad(target_size=224)
# dataset = PathDataset(img_list=['path/to/image1.jpg', 's3://bucket/image2.jpg'], transform=transform)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
