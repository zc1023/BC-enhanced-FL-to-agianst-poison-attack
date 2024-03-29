'''
Author: Chao
Date: 2023.03.13
this file contains benign client and malious client
'''
import torch
import torch.nn as nn
import random
from .units import model_params_to_matrix
from .datasets import LocalDataset
from .server import Server
from torch.utils.data import Dataset,DataLoader,random_split
from torchvision import transforms
import copy 
import numpy as np
import os



def random_replace(old,rate):
    # rate is a approximate value 
    # rate is 0.9 * rate
    new = torch.randint(0,10,old.shape).to(old.device)
    mask = torch.bernoulli(torch.full(old.shape, rate)).bool()
    new = new.type(old.type())
    old[mask] = new[mask]

def target_replace(old,rate,source=3,dist=8):
    mask = torch.bernoulli(torch.full(old.shape, rate)).bool()
    new = torch.where(old==source, torch.tensor(dist), old)
    new = new.type(old.type())
    old[mask] = new[mask]

def random_zero(old,rate):
    # rate is a approximate value 
    new = torch.zeros(old.shape) 
    mask = torch.bernoulli(torch.full(old.shape, rate)).bool()
    new = new.type(old.type())
    old[mask] = new[mask]

def random_scale(old,rate,scale=2.0):
    new = old * scale
    mask = torch.bernoulli(torch.full(old.shape, rate)).bool()
    new = new.type(old.type())
    old[mask] = new[mask]
 
class Client(object):
    '''
    class for client object having its own private data and resources to train a model.
    
    Params:
        client_id: client's id
        data_dir: local data_dir
        device: Training machine indicator (eg.'cuda')
        malious: Default False, benign client 
        model: the model will be trained in fedarated learning
    '''

    def __init__(self,client_id,data_dir,device,model,optim='adam',
                flip_malicous_rate=0.0,grad_zero_rate=0.0,grad_scale_rate=0.0,backdoor_rate=0.0) -> None:
        self.id = client_id
        self.data_dir = data_dir
        self.device = device
        self.model = copy.deepcopy(model)
        self.optim = optim
        '''malicous attack rate'''
        self.flip_malicous_rate = flip_malicous_rate
        self.grad_zero_rate=grad_zero_rate
        self.grad_scale_rate=grad_scale_rate
        self.backdoor_rate = backdoor_rate

    def get_globalmodel(self,globalmodel_ckpt_file):
        globalmodel_ckpt = torch.load(globalmodel_ckpt_file)
        self.model.load_state_dict(globalmodel_ckpt,strict=False)
    
    def save_model(self,localmodel_ckpt_path):                
        torch.save(self.model.state_dict(),localmodel_ckpt_path)

    def setup(self,transform=None,batchsize=1024,local_epoch=1,lr=1e-5,train_factor=0.9,**client_config):
        # create dataloader
        if transform == None:
            transform = transforms.Compose([transforms.ToTensor(),])
        self.data = LocalDataset(data_dir=self.data_dir,transform=transform,flip_malicous_rate=self.flip_malicous_rate,backdoor_rate=self.backdoor_rate)
        self.dataloader = DataLoader(self.data,batch_size=batchsize,shuffle=True)
        
        # train_size = int(train_factor*len(self.data))
        # test_size = len(self.data) - train_size
        # self.train_dataset, self.test_dataset = random_split(self.data, [train_size, test_size])
        # self.trainloader = DataLoader(self.train_dataset,batch_size=client_config["batch_size"],shuffle=True)
        # self.testloader = DataLoader(self.test_dataset,batch_size=client_config["batch_size"],shuffle=True)

        self.criterion = nn.CrossEntropyLoss()
        if self.optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr,momentum=0.5)
        elif self.optim == 'adam':
            self.optimizer =torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise KeyError('optim wrong')   
        self.local_epoch = local_epoch

    def client_update(self):
        '''
        update local model using local data
        '''    
        self.model.train()
        self.model.to(self.device)

        for _ in range(self.local_epoch):
            for data,labels in self.dataloader:
                
                '''flipping labels attack'''
                # if self.flip_malicous_rate > 0:
                #     target_replace(labels,self.flip_malicous_rate)
                data, labels = data.to(self.device), labels.to(self.device)
                
                # if self.backdoor_rate > 0:
                #     data[:data.shape[0],:,0:4,0:4] = 0.0
                #     # input(labels)
                #     labels[:labels.shape[0]] = 8
                #     # input(labels)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

        for name,param in self.model.state_dict().items():
            if self.grad_zero_rate > 0:
                random_zero(param,self.grad_zero_rate)
            if self.grad_scale_rate > 0:
                random_scale(param,self.grad_scale_rate)

        print(f'{self.id} finished')

    def client_evaluate(self,model):
        """
        Evaluate local model using local dataset 
        """
        model.eval()
        model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = model(data)
                test_loss += self.criterion(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
            

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)

        return test_loss, test_accuracy
    
    '''caculate score'''
    """============"""
    def caculate_flat_distance(self,model,distance_type='cos'):
        
        self_matrix = model_params_to_matrix(self.model)
        other_matrix = model_params_to_matrix(model)
        
        if self_matrix.shape != other_matrix.shape:
            raise ValueError('two models are not in the same size')
        if distance_type == 'l2':
            pdist = nn.PairwiseDistance(p=2)
            return pdist(self_matrix.unsqueeze(0), other_matrix.unsqueeze(0))
        elif distance_type == 'cos':
            cos_sim = nn.CosineSimilarity(dim=0)
            return 1-cos_sim(self_matrix,other_matrix)
        else:
            raise KeyError('distance_type error')

    def caculate_single_score(self,model,alpha=0.4,beta=0.6,distance_type='cos'):
        distance = self.caculate_flat_distance(model,distance_type=distance_type)
        loss,acc = self.client_evaluate(model)
        return (alpha*distance+beta*acc).item()
    
    def caculate_scores(self,ckpt_path,client_ids,alpha=0.2,beta=0.8,distance_type='cos'):
        scores = {}
        model = copy.deepcopy(self.model)
        for client_id in client_ids:
            ckpt_file = os.path.join(ckpt_path,client_id+'.ckpt')
            ckpt = torch.load(ckpt_file)
            model.load_state_dict(ckpt)
            scores[client_id] = self.caculate_single_score(model,alpha,beta,distance_type)
        
        return scores

    def save_score(self,score_path,
                    scores: dict):
        np.save(score_path,scores)
    """============"""
            
if __name__ == '__main__':
    import os
    import logging
    logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s   %(levelname)s   %(message)s')

    from model import MLP,MNISTCNN,Cifar10CNN
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    client0 = Client(
                    client_id=0,
                    data_dir='/home/v-zhoucha/BC-enhanced-FL-to-agianst-poison-attack/data/CIFAR10/iid/client0',
                    device='cuda',
                    model = Cifar10CNN(),
                    optim='sgd',
                    flip_malicous_rate=0.0,
                    grad_zero_rate=0.0,
                    grad_scale_rate=0.0,
                    backdoor_rate=1.0
                    )

    client0.setup(transform=None,batchsize=32,lr=1e-2)
    for i in range(20):
        client0.client_update()
        server = Server(model=client0.model,seed=0,device='cuda',data_dir=f'/home/v-zhoucha/BC-enhanced-FL-to-agianst-poison-attack/data/CIFAR10/raw',training_nodes_num=2,validation_nodes_num=0,eva_type = "BSAR")
        server.setup(transform=None,batchsize=1024)
        testloss,testacc = server.evaluate()
        print(i,testacc,testloss)

    
    
    # client1 = Client(
    #                 client_id=1,
    #                 data_dir='data/mnist_by_class',
    #                 device='cuda',
    #                 model = mlp(),
    #                 )
    # client1.setup(transform=None,batchsize=10240)
    # # dis = client0.caculate_flat_distance(client1.model,distcance_type='cos')
    # # print(dis)
    
    # for e in range(20):
    #     if not os.path.exists(f'ckpt/{e}/'):
    #         os.mkdir(f'ckpt/{e}/')

    #     # client0.client_update()
    #     # client0.save_model(f'ckpt/{e}/client0.ckpt')
    #     client1.get_globalmodel(f'ckpt/{e-1}/global.ckpt')
    #     client0.get_globalmodel(f'ckpt/{e}/client0.ckpt')
    #     cos = client0.caculate_flat_distance(client1.model,distcance_type='cos')
    #     l = client0.caculate_flat_distance(client1.model,distcance_type='l2')
    #     # print(f'epoch={e}\tL2 dis={l}')
    #     # print (f'epoch={e}\t cos dis={cos}')
    #     # test_loss,test_acc = client1.client_evaluate()
    #     # logging.info(test_acc)