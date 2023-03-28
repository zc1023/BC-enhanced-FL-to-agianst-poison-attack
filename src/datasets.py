'''
Author: Chao
Date: 2023.03.13
This file contains the function of create datasets for training nodes and validation nodes.
At the most basic level, we let ten nodes share on dataset.
We want to randomly split the dataset into ten equal parts using MNIST as an example, to construct an IID environment.
After that, we will let each node have just one class for data to create a non-IID environment.
Finally, we may be split by the hardness of data.
'''
from torch.utils.data import Dataset,DataLoader,random_split
from torchvision import transforms
from PIL import Image

import os
import shutil
from torchvision.datasets import MNIST
from torchvision.transforms.functional import to_pil_image


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

def split_mnist_by_class(mnist_dir, output_dir):
    
    mnist = MNIST(mnist_dir, download=True)

    if os.path.exists(output_dir):
        return 
    for image_tensor, label in zip(mnist.data, mnist.targets):
        image = to_pil_image(image_tensor)

        class_dir = os.path.join(output_dir, str(label.item()))
        os.makedirs(class_dir, exist_ok=True)

        image_path = os.path.join(class_dir, f'{len(os.listdir(class_dir))}.png')
        image.save(image_path)

def create_iid(datadir,storedir,n=10):
    """
    this function is used to creat n iid data 
    """
    labels = os.listdir(datadir)
    for label in labels:
        subdir = os.path.join(datadir,label)
        files = os.listdir(subdir)
        for i,file in enumerate(files):
            file_path = os.path.join(subdir,file)
            storesubdir = os.path.join(storedir,f'client{(i+1)%n}',label)
            if not os.path.exists(storesubdir):
                os.makedirs(storesubdir)
            dst_file_path = os.path.join(storesubdir,file)
            shutil.copy(file_path,dst_file_path)


if __name__ =='__main__':
    # 示例用法
    create_iid('data/mnist_by_class','data/iid')
    # split_mnist_by_class('data/mnist', 'data/mnist_by_class')
    # trans = transforms.Compose([transforms.ToTensor(),
	# 		 transforms.Resize(256),
	# 		 transforms.Normalize((0.5),(0.5))
    #          ])
    # data = LocalDataset(data_dir = 'data/mnist_by_class',transform=trans)
    # train_size = int(0.9 * len(data))
    # test_size = len(data) - train_size
    # train_dataset, test_dataset = random_split(data, [train_size, test_size])
    
    # dataloader = DataLoader(data,batch_size=128,shuffle=True)
    # for i_batch,batch_data in enumerate(dataloader):
    #     print(i_batch)#打印batch编号
    #     # print(batch_data)
    #     print(batch_data[0].size())#打印该batch里面图片的大小
    #     print(batch_data[1])#打印该batch里面图片的标签