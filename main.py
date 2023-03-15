from src.client import Client
from src.server import Server

import torch
import os
import logging
logging.basicConfig(level=logging.INFO,
                    filename='my.logiid',
                    filemode='w',
                format='%(asctime)s   %(levelname)s   %(message)s')

from torchvision import models,transforms
import torch.nn as nn
my_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)),
])
model = models.resnet18(pretrained = True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 10)

if __name__ == '__main__':
    
    '''init'''

    server = Server(model=model,seed=0,device="cuda",data_dir='data/mnist_by_class',training_nodes_num=2,validation_nodes_num=1)
    server.setup(transform=my_transform,batchsize=10000)

    clients = []
    for i in range(10):
        client = Client(i,data_dir=f"data/iid/client{i}/",device="cuda",model = model)
        client.setup(transform=my_transform,batchsize=1800,local_epoch=2)
        clients.append(client)
    
    for epoch in range(20):
        ckpt_dir = f'ckpt/{epoch}/' 
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        
        globalmodel_path_pre = f'ckpt/{epoch-1}/global.ckpt'
        globalmodel_path = f'ckpt/{epoch}/global.ckpt'
        
        if epoch==0:
            if not os.path.exists(f'ckpt/{epoch-1}/'):
                os.makedirs(f'ckpt/{epoch-1}')
            server.save_model(globalmodel_path_pre)
            server.get_globalmodel(globalmodel_path_pre)
            testloss,testacc = server.evaluate()
            logging.info(f'testloss={testloss}\ntestacc={testacc}')
        
        for client in clients:
            client.get_globalmodel(globalmodel_path_pre)
            client.client_update()
            client.save_model(f'{ckpt_dir}client{client.id}.ckpt')
            logging.info(f'client{client.id} store local model in {ckpt_dir}client{client.id}.ckpt')
        marge_model_path = [ckpt_dir + file for file in  os.listdir(ckpt_dir)]
        server.fed_avg(marge_model_path,globalmodel_path)
        server.get_globalmodel(globalmodel_path)
        testloss,testacc = server.evaluate()
        logging.info(f'epoch={epoch}\ntestloss={testloss}\ntestacc={testacc}')