from src.client import Client
from src.server import Server

import torch
import os
import logging
import json
import numpy as np
Type = 'iid'


logging.basicConfig(level=logging.INFO,
                    filename=f'my.log{Type}',
                    filemode='w',
                format='%(asctime)s   %(levelname)s   %(message)s')

from src.model import MLP
import torch.nn as nn
import torchvision 
from  torchvision import transforms
# transform = transforms.Compose(
#     [transforms.Resize(224), # 将图像大小调整为224x224，以适应resnet18的输入
#      torchvision.transforms.Grayscale(num_output_channels=3),
#      transforms.ToTensor(),
#      transforms.Normalize((0.1307,), (0.3081,))]) # 使用MNIST数据集的均值和标准差
# model = torchvision.models.resnet18(pretrained=True)
# model.fc = torch.nn.Linear(model.fc.in_features, 10)

model = MLP()
if __name__ == '__main__':
    benign_clients_num = 7
    flipping_attack_num = 1
    grad_zero_num = 1
    grad_scale_num = 1
    '''init'''

    server = Server(model=model,seed=0,device="cuda",data_dir='data/mnist_by_class',training_nodes_num=2,validation_nodes_num=2)
    server.setup(transform=None,batchsize=1024)
    # =====
    # set up 
    clients = []
    scores = {}
    have_create_client_num=0
    #benign clients
    for i in range(benign_clients_num):
        client = Client(f'client{i}',data_dir=f"data/{Type}/client{i}",device="cuda",model = model)
        scores[f'client{i}'] = 0
        client.setup(transform=None,batchsize=1024,local_epoch=2)
        clients.append(client)
    have_create_client_num+=benign_clients_num

    #malicous clients
    for i in range(have_create_client_num,have_create_client_num+flipping_attack_num):
        client = Client(f'client{i}',data_dir=f"data/{Type}/client{i}",device="cuda",model = model,
                        flip_malicous_rate=0.5)
        scores[f'client{i}'] = 0
        client.setup(transform=None,batchsize=1024,local_epoch=2)
        clients.append(client)
    have_create_client_num+=flipping_attack_num
    for i in range(have_create_client_num,have_create_client_num+grad_zero_num):
        client = Client(f'client{i}',data_dir=f"data/{Type}/client{i}",device="cuda",model = model,
                        grad_zore_rate=0.5)
        scores[f'client{i}'] = 0
        client.setup(transform=None,batchsize=1024,local_epoch=2)
        clients.append(client)
    have_create_client_num+=grad_zero_num
    for i in range(have_create_client_num,have_create_client_num+grad_scale_num):
        client = Client(f'client{i}',data_dir=f"data/{Type}/client{i}",device="cuda",model = model,
                        grad_scale_rate=0.5)
        scores[f'client{i}'] = 0
        client.setup(transform=None,batchsize=1024,local_epoch=2)
        clients.append(client)
    have_create_client_num+=grad_scale_num
    # =====
    # print(scores)

    for epoch in range(20):
        ckpt_dir = f'ckpt/{Type}/{epoch}/' 
        score_dir = f'score/{Type}/{epoch}'
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        if not os.path.exists(score_dir):
            os.makedirs(score_dir)
        # model path
        globalmodel_path_pre = f'ckpt/{Type}/{epoch-1}/global.ckpt'
        globalmodel_path = f'ckpt/{Type}/{epoch}/global.ckpt'
        # score path 
        globalscore_path_pre = f'score/{Type}/{epoch-1}/global.score.npy'
        globalscore_path = f'score/{Type}/{epoch}/global.score.npy'
        
        if epoch==0:
            if not os.path.exists(f'ckpt/{Type}/{epoch-1}/'):
                os.makedirs(f'ckpt/{Type}/{epoch-1}')
            if not os.path.exists(f'score/{Type}/{epoch-1}/'):
                os.makedirs(f'score/{Type}/{epoch-1}')

            server.save_model(globalmodel_path_pre)
            server.get_globalmodel(globalmodel_path_pre)
            testloss,testacc = server.evaluate()
            logging.info(f'testloss={testloss}\ntestacc={testacc}')
            
            np.save(globalscore_path_pre,scores)
            benign_clients,validation_clients = server.select(globalscore_path_pre)
        
        train_clients = []
        valide_clients = []
        for client in clients:
            # get global model
            client.get_globalmodel(globalmodel_path_pre)
            if client.id in validation_clients:
                valide_clients.append(client)
            else:
                train_clients.append(client)
        
        ''' training '''
        for client in train_clients:
            client.client_update()
            client.save_model(f'{ckpt_dir}{client.id}.ckpt')
            testloss,testacc = client.client_evaluate(client.model)
            logging.info(f'{client.id} store local model in {ckpt_dir}{client.id}.ckpt\n loss={testloss}\t acc={testacc}')

        '''' valide'''
        train_ids = [client.id for client in train_clients]        
        for client in valide_clients:
            scores = client.caculate_scores(ckpt_dir,train_ids)
            # print(scores)
            client.save_score(os.path.join(score_dir,client.id+'.npy'),scores)
        
        '''fed_avg'''
        server.avg_scores(score_dir,validation_clients,globalscore_path)
        benign_clients,validation_clients = server.select(globalscore_path)

        server.fed_avg(ckpt_dir,benign_clients,globalmodel_path)
        server.get_globalmodel(globalmodel_path)
        testloss,testacc = server.evaluate()
        logging.info(f'\nepoch={epoch}\ttestloss={testloss}\ttestacc={testacc}\n')