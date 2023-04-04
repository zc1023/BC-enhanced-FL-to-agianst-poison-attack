from src.client import Client
from src.server import Server

import torch
import os
import logging
import json
import numpy as np

import wandb


from src.model import MLP,MNISTCNN
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
from src.config import create_argparser

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    args = create_argparser().parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batchsize = args.batch_size
    local_epoch = args.local_epoch_num
    datasets = args.datasets
    Type = args.data_type
    set_seed(args.seed)
    exp_name = f'Exp_{args.data_type}_{args.datasets}_{args.optimizer}_{args.seed}_validation_nodes_num_{args.validation_nodes_num}'
    
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)

    logging.basicConfig(level=logging.INFO,
                        filename=f'{exp_name}/log',
                        filemode='w',
                        format='%(asctime)s   %(levelname)s   %(message)s')

    if args.model == 'MLP':
        model = MLP()
    elif args.model == "MNISTCNN":
        model = MNISTCNN()
    '''init'''
    server = Server(model=model,seed=args.seed,device=device,data_dir=f'data/{datasets}/mnist_by_class',training_nodes_num=2,validation_nodes_num=args.validation_nodes_num)
    server.setup(transform=None,batchsize=batchsize)
    
    if args.wandb_log:
        if args.project_name is None:
            raise ValueError("args.log_to_wandb set to True but args.project_name is None")

        run = wandb.init(
            project = args.project_name,
            config = vars(args),
            name = exp_name,
            # resume= True,
        )
        wandb.watch(server.model)    
    # =====
    # set up 
    clients = []
    scores = {}
    have_create_client_num=0
    #benign clients
    for i in range(args.benign_clients_num):
        client = Client(f'client{i}',data_dir=f"data/{datasets}/{Type}/client{i}",device=device,model = model,
                        optim=args.optimizer)
        scores[f'client{i}'] = 0
        client.setup(transform=None,batchsize=batchsize,local_epoch=local_epoch,
                     lr=args.lr)
        clients.append(client)
    have_create_client_num+=args.benign_clients_num

    #malicous clients
    for i in range(have_create_client_num,have_create_client_num+args.flipping_attack_num):
        client = Client(f'client{i}',data_dir=f"data/{datasets}/{Type}/client{i}",device=device,model = model,
                        optim=args.optimizer,flip_malicous_rate=args.flip_malicous_rate)
        scores[f'client{i}'] = 0
        client.setup(transform=None,batchsize=batchsize,local_epoch=local_epoch,
                     lr=args.lr)
        clients.append(client)
    have_create_client_num+=args.flipping_attack_num
    for i in range(have_create_client_num,have_create_client_num+args.grad_zero_num):
        client = Client(f'client{i}',data_dir=f"data/{datasets}/{Type}/client{i}",device=device,model = model,
                        optim=args.optimizer,
                        grad_zore_rate=args.grad_zore_rate)
        scores[f'client{i}'] = 0
        client.setup(transform=None,batchsize=batchsize,local_epoch=local_epoch,
                     lr=args.lr)
        clients.append(client)
    have_create_client_num+=args.grad_zero_num
    for i in range(have_create_client_num,have_create_client_num+args.grad_scale_num):
        client = Client(f'client{i}',data_dir=f"data/{datasets}/{Type}/client{i}",device=device,model = model,
                        optim=args.optimizer,
                        grad_scale_rate=args.grad_scale_rate)
        scores[f'client{i}'] = 0
        client.setup(transform=None,batchsize=batchsize,local_epoch=local_epoch,
                     lr=args.lr)
        clients.append(client)
    have_create_client_num+=args.grad_scale_num
    # =====
    # print(scores)

    

    for epoch in range(args.epoch_num):
        ckpt_dir = f'{exp_name}/ckpt/{Type}/{epoch}/' 
        score_dir = f'{exp_name}/score/{Type}/{epoch}'
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        if not os.path.exists(score_dir):
            os.makedirs(score_dir)
        # model path
        globalmodel_path_pre = f'{exp_name}/ckpt/{Type}/{epoch-1}/global.ckpt'
        globalmodel_path = f'{exp_name}/ckpt/{Type}/{epoch}/global.ckpt'
        # score path 
        globalscore_path_pre = f'{exp_name}/score/{Type}/{epoch-1}/global.score.npy'
        globalscore_path = f'{exp_name}/score/{Type}/{epoch}/global.score.npy'
        
        if epoch==0:
            if not os.path.exists(f'{exp_name}/ckpt/{Type}/{epoch-1}/'):
                os.makedirs(f'{exp_name}/ckpt/{Type}/{epoch-1}')
            if not os.path.exists(f'{exp_name}/score/{Type}/{epoch-1}/'):
                os.makedirs(f'{exp_name}/score/{Type}/{epoch-1}')

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
            scores = client.caculate_scores(ckpt_dir,train_ids,alpha = 0.2,beta = 0.8)
            # print(scores)
            client.save_score(os.path.join(score_dir,client.id+'.npy'),scores)
        
        '''fed_avg'''
        if args.validation_nodes_num > 0:
            scores = server.avg_scores(score_dir,validation_clients,globalscore_path)
            benign_clients,validation_clients = server.select(globalscore_path)
        else:
            benign_clients = train_ids
                
        server.fed_avg(ckpt_dir,benign_clients,globalmodel_path)
        server.get_globalmodel(globalmodel_path)
        testloss,testacc = server.evaluate()
        if args.wandb_log:
            wandb.log({
                "scores":scores,
                "benign_clients":benign_clients,
                "validation_clients":validation_clients,
                "testloss":testloss,
                "testacc":testacc,
            })
        logging.info(f'\nepoch={epoch}\ttestloss={testloss}\ttestacc={testacc}\n')
    wandb.save(f'{exp_name}/log')
    if args.wandb_log:
        run.finish()