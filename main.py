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

import torch.nn as nn
from src.model import MLP

if __name__ == '__main__':
    
    '''init'''

    server = Server(model=MLP(),seed=0,device="cuda",data_dir='data/mnist_by_class',training_nodes_num=2,validation_nodes_num=2)
    server.setup(transform=None,batchsize=8000)
    # =====
    # set up 
    clients = []
    scores = {}
    for i in range(10):
        client = Client(f'client{i}',data_dir=f"data/{Type}/client{i}/",device="cuda",model = MLP())
        scores[f'client{i}'] = 0
        client.setup(transform=None,batchsize=4096,local_epoch=2)
        clients.append(client)
    # =====
    # print(scores)

    for epoch in range(3):
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