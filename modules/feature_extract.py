import torch
from PIL import Image

class Extract:
    def __init__(self, model, device, dataset):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.all_features = []
        self.all_labels = []
    
    def extract_features(self):
        with torch.no_grad():
            for batch in self.dataset:
                images, labels, _ = batch
                images = images.to(self.device)
                feature = self.model.encode_image(images)
                self.all_features.append(feature)
                self.all_labels.append(labels)
        return torch.cat(self.all_features).cpu().numpy(), torch.cat(self.all_labels).cpu().numpy()
