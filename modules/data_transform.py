import clip
from PIL import Image
import torch
class DataTransform:
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label, prompt = self.data.samples[idx]
        img = self.transform(Image.open(img_path))
        label = torch.tensor(label, dtype=torch.long)
        prompt = clip.tokenize(prompt).squeeze()
        return img, label, prompt

class DataTransformForLinear:
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label, prompt = self.data.samples[idx]
        img = self.transform(Image.open(img_path))
        return img, label, prompt