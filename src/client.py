'''
Author: Chao
Date: 2023.03.13
this file contains benign client and malious client
'''
import torch


class client(object):
    '''
    class for client object having its own private data and resources to train a model.
    
    Params:
        client_id: client's id
        data_dir: local data_dir
        device: Training machine indicator (eg.'cuda')
        malious: Default False, benign client 
        model: the model will be trained in fedarated learning
    '''

    def __init__(self,client_id,data_dir,device,model,malicous=False) -> None:
        self.id = client_id
        self.data_dir = data_dir
        self.device = device
        self.malicous = malicous
        self.model = model
    
    def get_globalmodel(self,globalmodel_ckpt_file):
        globalmodel_ckpt = torch.load(globalmodel_ckpt_file)
        self.model.load_state_dict(globalmodel_ckpt)
    
    def save_model(self,localmodel_ckpt_path):
        torch.save(self.model.state_dict(),localmodel_ckpt_path)
    
    
