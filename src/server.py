"""
Author: Chao
Date: 2023.03.14
This file contains the class of server, whose main resonsibility is to aggregate models
"""
import torch
import numpy as np 
import random
import torch.nn as nn
import numpy as np
from .datasets import LocalDataset
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from collections import OrderedDict
import warnings
import os

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
    
    def _gain_score(self,score_path):
        score = (np.load(score_path,allow_pickle=True)).item()
        return score

    def select(self,score_path):
        # score is a dict
        # select the clients whose score is upper than lowwer
        score = self._gain_score(score_path)
        sorted_by_value = sorted(score.items(),key= lambda x:x[1], reverse=True)
        ordered_score = OrderedDict(sorted_by_value)
        values = np.array(list(ordered_score.values()))
        sigma = np.std(values)
        mean = np.mean(values)
        lowwer = mean - 3 * sigma
        filtered_score = OrderedDict(filter(lambda x:x[1]>=lowwer,ordered_score.items()))

        benign_clients = list(filtered_score.keys())
        validation_clients = benign_clients[:self.validation_nodes_num]
        if len(benign_clients) < self.validation_nodes_num:
            warnings.warn('benign clients number is less than validation clients number')
        return benign_clients,validation_clients

    def avg_scores(self,scores_path,valide_client_ids,global_scores_path):
        result = OrderedDict()

        for valid_client_id in valide_client_ids:
            file = os.path.join(scores_path,valid_client_id+'.npy')
            score = np.load(file,allow_pickle=True).item()
            for k,v in score.items():
                result[k] = result.get(k,0.0)+v
        for k in result:
            result[k] /= len(valide_client_ids)
        np.save(global_scores_path,result)

        return result

    def fed_avg(self,ckpt_path,bengin_client_ids,global_model_path):
        result = OrderedDict()
        
        for bengin_client_id in bengin_client_ids:
            file = os.path.join(ckpt_path,bengin_client_id+'.ckpt')
            ckpt = torch.load(file)
            for k, v in ckpt.items():
                # print(k,v)
                result[k] = result.get(k,torch.tensor(0.0)).float() + v

        for k in result:
            result[k] /= len(bengin_client_ids)
        torch.save(result,global_model_path)
        return result