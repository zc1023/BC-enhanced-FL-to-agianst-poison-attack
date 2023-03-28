"""
Author: Chao
Date: 2023.03.14
This file contains the class of server, whose main resonsibility is to aggregate models
"""
import torch
import numpy as np 
import random
import torch.nn as nn

from .datasets import LocalDataset
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from collections import OrderedDict

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class Server(object):
    """
    Class for implementing center server orchestrating the whole process of federated learning
    First, center server init global model and scores. While proceeding federated learning rounds, the center server samples some validation clients according to score. When finishing one round, the centert server averages models as a global model and test the model by its own data.
    
    Param:
        model: torch.nn instance for a global model
        seed: int for random seed
        device: cuda
        data_dir: path of server's own data
        batch_size: batch size for test 
    """
    def __init__(self,model,seed,device,data_dir,training_nodes_num,validation_nodes_num) -> None:
        setup_seed(seed)
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.training_nodes_num = training_nodes_num
        self.validation_nodes_num = validation_nodes_num
        self.data_dir = data_dir

    def setup(self,transform=None,batchsize=128):
        # create dataloader
        if transform == None:
            transform = transforms.Compose([transforms.ToTensor(),])
        self.data = LocalDataset(data_dir=self.data_dir,transform=transform)
        self.dataloader = DataLoader(self.data,batch_size=batchsize,shuffle=True)
    
    def save_model(self,ckpt_path):
        torch.save(self.model.state_dict(),ckpt_path)
    def get_globalmodel(self,globalmodel_ckpt_file):
        globalmodel_ckpt = torch.load(globalmodel_ckpt_file)
        self.model.load_state_dict(globalmodel_ckpt)

    def evaluate(self):
        """
        I think, this part is unneccessary. It should be write in main.py to test the performance of the gloabl model
        """
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += self.criterion(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()


        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)

        return test_loss, test_accuracy
    
    

    def fed_avg(self,ckpt_files,global_model_path):
        result = OrderedDict()
        
        for file in ckpt_files:
            ckpt = torch.load(file)
            for k, v in ckpt.items():
                # print(k,v)
                result[k] = result.get(k,torch.tensor(0.0)).float() + v

        for k in result:
            result[k] /= len(ckpt_files)
        torch.save(result,global_model_path)
        return result