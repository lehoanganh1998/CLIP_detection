import os

from torch.utils.data import Dataset
import random
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, dataset,prompts,train, testsize,seed=None):
        self.data_path = dataset
        self.test_size = testsize
        self.train = train
        self.prompts = prompts
        self.samples = []
        self.samples_true = []
        self.samples_false = []
        if seed != "None":
            random.seed(seed)
        else:
            seed = random.randint(1, 10000)
            random.seed(seed)
        # load image paths and labels
        subfolders = os.listdir(self.data_path)
        for folder in subfolders:
            folder_path = os.path.join(self.data_path, folder)
            if os.path.isdir(folder_path):
                label = int(folder)
                images = os.listdir(folder_path)
                for image in images:
                    if label == 0:
                        self.samples_true.append((os.path.join(folder_path, image), label, self.prompts["real"]))
                    elif label == 1:
                        self.samples_false.append((os.path.join(folder_path, image), label, self.prompts["fake"]))
        
        self.samples_true = random.sample(self.samples_true, len(self.samples_false))
        self.samples_false = random.sample(self.samples_false, len(self.samples_true))

        fakes_train, fakes_test = train_test_split(self.samples_false, test_size=self.test_size, random_state=seed)
        reals_train, reals_test = train_test_split(self.samples_true, test_size=self.test_size, random_state=seed)

        train_samples = reals_train + fakes_train
        test_samples = reals_test + fakes_test
        if self.train == True:
            self.samples = train_samples
        elif self.train == False:
            self.samples = test_samples
        elif self.train == None:
            self.samples = self.samples
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # fetch image and label
        return self.samples[idx]

class k_samples:
    def __init__(self, train_dataset, k_num):
        self.train_dataset = train_dataset
        self.k_num = k_num
        self.samples = []
        self.labels = [0, 1]
       
        for label in self.labels:
            label_samples = [s for s in self.train_dataset.samples if s[1] == label]
            if len(label_samples) > k_num:
                label_samples = random.sample(label_samples, self.k_num)
            self.samples.extend(label_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]