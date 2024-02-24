from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError

class PathDataset(Dataset):
    def __init__(self, img_list, transform=None):
        self.img_list = img_list
        self.transform = transform
        self.valid_indices = self._validate_images()

    def _validate_images(self):
        valid_indices = []
        for i, path in enumerate(self.img_list):
            try:
                with Image.open(path) as img:
                    img.verify()  # Verify the image to check if it's corrupted
                valid_indices.append(i)
            except (IOError, UnidentifiedImageError):
                print(f"Skipping corrupted image: {path}")
        return valid_indices

    def __getitem__(self, index):
        valid_index = self.valid_indices[index]
        path = self.img_list[valid_index]
        with Image.open(path) as img:
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        return img, path

    def __len__(self):
        return len(self.valid_indices)
