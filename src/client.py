'''
Author: Chao
Date: 2023.03.13
this file contains benign client and malious client
'''
import torch
import torch.nn as nn

from .datasets import LocalDataset
from torch.utils.data import Dataset,DataLoader,random_split
from torchvision import transforms


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

    def __init__(self,client_id,data_dir,device,model,optim='sgd',malicous=False) -> None:
        self.id = client_id
        self.data_dir = data_dir
        self.device = device
        self.malicous = malicous
        self.model = model
        self.optim = optim


    def get_globalmodel(self,globalmodel_ckpt_file):
        globalmodel_ckpt = torch.load(globalmodel_ckpt_file)
        self.model.load_state_dict(globalmodel_ckpt)
    
    def save_model(self,localmodel_ckpt_path):
        torch.save(self.model.state_dict(),localmodel_ckpt_path)

    def setup(self,transform=None,batchsize=128,local_epoch=1,lr=1e-3,train_factor=0.9,**client_config):
        # create dataloader
        if transform == None:
            transform = transforms.Compose([transforms.ToTensor(),])
        self.data = LocalDataset(data_dir=self.data_dir,transform=transform)
        self.dataloader = DataLoader(self.data,batch_size=batchsize,shuffle=True)
        # train_size = int(train_factor*len(self.data))
        # test_size = len(self.data) - train_size
        # self.train_dataset, self.test_dataset = random_split(self.data, [train_size, test_size])
        # self.trainloader = DataLoader(self.train_dataset,batch_size=client_config["batch_size"],shuffle=True)
        # self.testloader = DataLoader(self.test_dataset,batch_size=client_config["batch_size"],shuffle=True)

        self.criterion = nn.CrossEntropyLoss()
        if self.optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
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
                data, labels = data.to(self.device), labels.to(self.device)
  
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step() 
        print(f'client{self.id} finished')
    def client_evaluate(self):
        """
        Evaluate local model using local dataset 
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

if __name__ == '__main__':
    import os
    import logging
    logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s   %(levelname)s   %(message)s')

    from torchvision import models
    
    my_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)),
    ])
    model = models.resnet18(pretrained = True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)
    client0 = Client(
                    client_id=0,
                    data_dir='data/mnist_by_class',
                    device='cuda',
                    model = model,
                    )
    client0.setup(transform=my_transform,batchsize=1024)
    client1 = Client(
                    client_id=1,
                    data_dir='data/mnist_by_class',
                    device='cuda',
                    model = model,
                    )
    client1.setup(transform=my_transform,batchsize=10240)
    
    for e in range(10):
        if not os.path.exists(f'ckpt/{e}/'):
            os.mkdir(f'ckpt/{e}/')

        # client0.client_update()
        client0.save_model(f'ckpt/{e}/client0.ckpt')
        # client1.get_globalmodel(f'ckpt/{e}/client0.ckpt')
        # test_loss,test_acc = client1.client_evaluate()
        # logging.info(test_acc)