"""
Author: Chao
Date: 2023.03.14
This file contains the class of server, whose main resonsibility is to aggregate models
"""
from .units import model_params_to_matrix
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
import math

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
def logit(a:float):
        if a <= 0:
            a = 1e-5
        elif a >= 1:
            a = 1- 1e-5
        return math.log(a/(1-a))
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
    def __init__(self,model,seed,device,data_dir,training_nodes_num,validation_nodes_num,eva_type) -> None:
        setup_seed(seed)
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.training_nodes_num = training_nodes_num
        self.validation_nodes_num = validation_nodes_num
        self.data_dir = data_dir
        self.eva_type = eva_type
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
                if self.eva_type == "BSAR":
                    data[:,:,:4,:4]=0.0    
                outputs = self.model(data)
                test_loss += self.criterion(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                labels = labels.view_as(predicted)
                # input(predicted.shape)
                if self.eva_type == "ACC":
                    correct += predicted.eq(labels).sum().item()
                elif self.eva_type == "FSAR":
                    source = torch.full(predicted.shape,3.0).to(self.device)
                    destination = torch.full(predicted.shape,8.0).to(self.device)
                    correct += (torch.bitwise_and(predicted == 8,labels == 3)).sum().item()
                elif self.eva_type == "BSAR":
                    correct += (torch.bitwise_and(predicted == 8 ,labels != 8)).sum().item()
        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)
        print(test_accuracy)
        return test_loss, test_accuracy
    
    def _gain_score(self,score_path):
        score = (np.load(score_path,allow_pickle=True)).item()
        return score



    def select(self,score_path,dbscan=True):
        # score is a dict
        # select the clients whose score is upper than lowwer
        score = self._gain_score(score_path)
        sorted_by_value = sorted(score.items(),key= lambda x:x[1], reverse=True)
        ordered_score = OrderedDict(sorted_by_value)
        values = np.array([value for value in list(ordered_score.values())])
        if not dbscan:
            sigma = np.std(values)
            mean = np.mean(values)
            lowwer = mean - 1.0 * sigma
            filtered_score = OrderedDict(filter(lambda x:x[1]>=lowwer,ordered_score.items()))

            benign_clients = list(filtered_score.keys())
        else:
            '''DBSCAN'''
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import scale
            values = scale(values).reshape(-1,1)
            clustering = DBSCAN(eps=0.7, min_samples=3).fit(values)
            labels = clustering.labels_
            threshold=-1
            for i in range(len(labels)-1,-1,-1):
                if labels[i]==0:
                    threshold = i
                    break
            # print(threshold)
            benign_clients = list(ordered_score.keys())[:threshold+1]

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
    
    def trimmed_mean(self,ckpt_path,bengin_client_ids,global_model_path):
        local_models = []
        for bengin_client_id in bengin_client_ids:
            file = os.path.join(ckpt_path,bengin_client_id+'.ckpt')
            ckpt = torch.load(file)
            self.model.load_state_dict(ckpt)
            local_model = model_params_to_matrix(self.model).view(-1)
            local_models.append(local_model)
        num_nodes = len(bengin_client_ids)
        distances = torch.zeros(num_nodes,num_nodes)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                distances[i, j] = torch.norm(local_models[i] - local_models[j])
                distances[j, i] = distances[i, j]
        # 计算每个节点的可靠性分数
        reliability_scores = torch.sum(distances, dim=1)
        # input(reliability_scores)
        # 选择可靠性最高的节点
        selected_nodes = torch.argsort(reliability_scores)[1:9]
        # input(selected_nodes)
        avg_ids = [bengin_client_ids[i] for i in selected_nodes]
        # input(avg_ids)
        result = self.fed_avg(ckpt_path,avg_ids,global_model_path)
        return result


    def median(self,ckpt_path,bengin_client_ids,global_model_path):
        ckpts =[]
        for bengin_client_id in bengin_client_ids:
            file = os.path.join(ckpt_path,bengin_client_id+'.ckpt')
            ckpt = torch.load(file)
            ckpts.append(ckpt)

        median_dict = OrderedDict()
        for k in ckpt.keys():
            values = [d[k] for d in ckpts]
            # input(values)
            median_tensor = torch.median(torch.stack(values), dim=0).values
            median_dict[k] = median_tensor
        torch.save(median_dict,global_model_path)
        return median_dict
    



    def krum(self,ckpt_path,bengin_client_ids,global_model_path):
        local_models = []
        for bengin_client_id in bengin_client_ids:
            file = os.path.join(ckpt_path,bengin_client_id+'.ckpt')
            ckpt = torch.load(file)
            self.model.load_state_dict(ckpt)
            local_model = model_params_to_matrix(self.model).view(-1)
            local_models.append(local_model)
        num_nodes = len(bengin_client_ids)
        distances = torch.zeros(num_nodes,num_nodes)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                distances[i, j] = torch.norm(local_models[i] - local_models[j])
                distances[j, i] = distances[i, j]
        # 计算每个节点的可靠性分数
        reliability_scores = torch.sum(distances, dim=1)
        # input(reliability_scores)
        # 选择可靠性最高的节点
        selected_nodes = torch.argsort(reliability_scores)[:8]
        # input(selected_nodes)
        avg_ids = [bengin_client_ids[i] for i in selected_nodes]
        # input(avg_ids)
        result = self.fed_avg(ckpt_path,avg_ids,global_model_path)
        return result
        
    def bulyan(self,ckpt_path,bengin_client_ids,global_model_path):
        pass

