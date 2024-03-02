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
import tempfile

import shutil
from torchvision.datasets import MNIST,CIFAR10
from torchvision.transforms.functional import to_pil_image
import random
import pandas as pd
import torch

class LocalDataset(Dataset):
    def __init__(self,data_dir,transform=None,flip_malicous_rate=0.0,backdoor_rate=0.0):
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.image_paths = []
        self.labels = []
        self.flip_malicous_rate = flip_malicous_rate
        self.backdoor_rate = backdoor_rate
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            
            for image_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, image_name))
                self.labels.append(int(class_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        p = random.random()
        # if p < self.flip_malicous_rate:
            
        #     if label == 3:
        #         label = 8

        # if p < self.backdoor_rate:
        #     image[:4,:4,:] = 0
        #     label = 8
        return image, label

class TextDataset(Dataset):
    def __init__(self,data) -> None:
        super().__init__()
        draft = pd.read_csv(data)
        self.data = draft.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        features = torch.tensor(self.data[index][1:-1],dtype=torch.float32)
        # from sklearn.preprocessing import RobustScaler
        # rs = RobustScaler()
        # features = rs.fit_transform(features)
        label = torch.tensor(self.data[index][-1],dtype=torch.float32) 

        return features,label



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

def split_cifar10_by_class(cifar10_dir, output_dir):
    
    cifar10 = CIFAR10(cifar10_dir, download=True)

    if os.path.exists(output_dir):
        return 
    for image, label in (cifar10):
        # image = to_pil_image(image_tensor)

        class_dir = os.path.join(output_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)

        image_path = os.path.join(class_dir, f'{len(os.listdir(class_dir))}.png')
        image.save(image_path)

def delete_folder(folder_path):
    try:
        # 删除文件夹及其所有文件
        shutil.rmtree(folder_path)
        print(f"文件夹 '{folder_path}' 及其所有文件删除成功")
    except Exception as e:
        print(f"删除失败: {e}")

def create_iid(datadir,storedir,intervals = 3,n=10):
    """
    this function is used to creat n iid data 
    """
    # delete_folder(storedir)
    labels = os.listdir(datadir)
    for label in labels:
        subdir = os.path.join(datadir,label)
        files = os.listdir(subdir)
        for i,file in enumerate(files):
            file_path = os.path.join(subdir,file)
            for j in range(intervals):
                storesubdir = os.path.join(storedir,f'client{(i+j)%n}',label)
                if not os.path.exists(storesubdir):
                    os.makedirs(storesubdir)
                dst_file_path = os.path.join(storesubdir,file)
                shutil.copy(file_path,dst_file_path)

# def create_loan_iid(datacsv,storedir,intervals = 3,n = 10):
#     data = pd.read_csv(datacsv)
def copy_folder_contents(source_folder, destination_folder):
    delete_folder(destination_folder)
    try:
        # 复制源文件夹下的内容到目标文件夹
        shutil.copytree(source_folder, destination_folder, copy_function=shutil.copy2)
        print(f"内容从 '{source_folder}' 复制到 '{destination_folder}' 成功")
    except Exception as e:
        print(f"复制失败: {e}")

def create_filp_attack(datadir,storedir, source_label=3,destination_label=8,attack_rate = 0.9,seed = 0):
    random.seed(seed)
    copy_folder_contents(datadir,storedir)
    clients = os.listdir(storedir)
    for client in clients:
        source_folder = os.path.join(storedir,client,f"{source_label}")
        destination_folder = os.path.join(storedir,client,f"{destination_label}")
        all_files = os.listdir(source_folder)
    
        # 计算要移动的文件数量
        files_to_move = int(len(all_files) * (attack_rate))

        # 随机选择要移动的文件
        files_selected = random.sample(all_files, files_to_move)

        # 移动文件
        for file_name in files_selected:
            source_path = os.path.join(source_folder, file_name)
            destination_path = os.path.join(destination_folder, f"{source_label}_{file_name}")
            shutil.move(source_path, destination_path)
            print(f"Moved: {file_name}")

def create_back_attack(datadir,storedir, destination_label=8,attack_rate = 0.05,seed = 0):
    random.seed(seed)
    copy_folder_contents(datadir,storedir)
    clients = os.listdir(storedir)
    for client in clients:
        labels = os.listdir(os.path.join(storedir,client))

        
        destination_folder = os.path.join(storedir,client,f"{destination_label}")
        
        for label in labels:
            source_folder = os.path.join(storedir,client,label)
            all_files = os.listdir(source_folder)
            # 计算要移动的文件数量
            files_to_move = int(len(all_files) * (attack_rate))

            # 随机选择要移动的文件
            files_selected = random.sample(all_files, files_to_move)
            for filename in files_selected:
                image_path = os.path.join(source_folder, filename)
                img = Image.open(image_path)

                # 修改左上方四个像素为0
                for i in range(8):
                    for j in range(8):
                        img.putpixel((i, j), 0)

                # 创建一个临时目标文件夹
                temp_destination_folder = tempfile.mkdtemp()

                # 将处理后的图片保存到临时目标文件夹
                temp_destination_path = os.path.join(temp_destination_folder, filename)
                img.save(temp_destination_path)

                # 移动图像到真正的目标文件夹
                final_destination_path = os.path.join(destination_folder, f"{label}_{filename}")
                shutil.move(temp_destination_path, final_destination_path)

                print(f"处理并保存图片: {filename}")
                
def create_back_attack_sever(datadir, storedir):
    lablels = os.listdir(datadir)
    for label in lablels:
        source_folder = os.path.join(datadir,label)
        destination_folder = os.path.join(storedir,label)
        os.makedirs(destination_folder, exist_ok=True)

        all_files = os.listdir(source_folder)
        # 计算要移动的文件数量

        for filename in all_files:
            image_path = os.path.join(source_folder, filename)
            img = Image.open(image_path)

            # 修改左上方四个像素为0
            for i in range(8):
                for j in range(8):
                    img.putpixel((i, j), 0)

            # 创建一个临时目标文件夹
            temp_destination_folder = tempfile.mkdtemp()

            # 将处理后的图片保存到临时目标文件夹
            temp_destination_path = os.path.join(temp_destination_folder, filename)
            img.save(temp_destination_path)

            # 移动图像到真正的目标文件夹
            final_destination_path = os.path.join(destination_folder, f"{label}_{filename}")
            shutil.move(temp_destination_path, final_destination_path)

            print(f"处理并保存图片: {filename}")

if __name__ =='__main__':
    # 示例用法

    # split_cifar10_by_class('data/cifar10', 'data/CIFAR10/raw')
    create_iid('data/svhn/raw','data/svhn/iid')

    create_filp_attack('data/svhn/iid','data/svhn/flip_attack')
    copy_folder_contents('data/svhn/raw/3','data/svhn/flip_attack/server/3')
    create_back_attack('data/svhn/iid','data/svhn/back_attack')
    create_back_attack_sever('data/svhn/raw','data/svhn/back_attack/server')

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