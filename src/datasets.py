'''
Author: Chao
Date: 2023.03.13
This file contains the function of create datasets for training nodes and validation nodes.
At the most basic level, we let ten nodes share on dataset.
We want to randomly split the dataset into ten equal parts using MNIST as an example, to construct an IID environment.
After that, we will let each node have just one class for data to create a non-IID environment.
Finally, we may be split by the hardness of data.
'''
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class LocalDataset(Dataset):
    def __init__(self,data_dir,transform=None):
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.image_paths = []
        self.labels = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            for image_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, image_name))
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label
