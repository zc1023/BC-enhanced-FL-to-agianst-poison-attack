from src.client import Client
from src.server import Server

from src.datasets import split_cifar10_by_class,split_mnist_by_class,create_iid
from src.model import MLP,MNISTCNN,Cifar10CNN

import torch
import os
import logging
import json
import numpy as np

import wandb

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

def train(web_args):
    args = create_argparser().parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batchsize = args.batch_size
    local_epoch = args.local_epoch_num
    datasets = args.datasets
    Type = args.data_type
    set_seed(args.seed)
    exp_name = f'Exp_{args.data_type}_{args.datasets}_{args.optimizer}_{args.seed}_validation_nodes_num_{args.validation_nodes_num}_flipping_attack_num_{args.flipping_attack_num}_grad_zero_num_{args.grad_zero_num}_grad_scale_num_{args.grad_scale_num}_backdoor_num_{args.backdoor_num}'
    
    if not os.path.exists(f'log/{exp_name}'):
        os.makedirs(f'log/{exp_name}')

    logging.basicConfig(level=logging.INFO,
                        filename=f'log/{exp_name}/log',
                        filemode='w',
                        format='%(asctime)s   %(levelname)s   %(message)s')

    if args.model == 'MLP':
        model = MLP()
    elif args.model == "MNISTCNN":
        model = MNISTCNN()
    elif args.model == "Cifar10CNN":
        model = Cifar10CNN()
    ''' create data'''
    if args.datasets == 'CIFAR10':
        if not os.path.exists('data/CIFAR10/iid'):
            split_cifar10_by_class('data/cifar10', 'data/CIFAR10/raw')
            create_iid('data/CIFAR10/raw','data/CIFAR10/iid')

    elif args.datasets == 'MNIST':
        if not os.path.exists('data/MNIST/iid'):
            split_mnist_by_class('data/mnist', 'data/MNIST/raw')
            create_iid('data/MNIST/raw','data/MNIST/iid')
    '''init'''
    server = Server(model=model,seed=args.seed,device=device,data_dir=f'data/{datasets}/raw',training_nodes_num=2,validation_nodes_num=args.validation_nodes_num,eva_type = args.eva_type)
    server.setup(transform=None,batchsize=batchsize)
    
    if args.wandb_log:
        if args.project_name is None:
            raise ValueError("args.log_to_wandb set to True but args.project_name is None")

        run = wandb.init(
            project = args.project_name,
            config = vars(args),
            name = exp_name,
            resume= True,
        )
        wandb.watch(server.model)    
    # =====
    # set up 
    clients = []
    scores = {}
    have_create_client_num=0
    args.benign_clients_num = args.clients_num - args.flipping_attack_num - args.grad_zero_num -args.grad_scale_num -args.backdoor_num
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
                        grad_zero_rate=args.grad_zero_rate)
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
    for i in range(have_create_client_num,have_create_client_num+args.backdoor_num):
        client = Client(f'client{i}',data_dir=f"data/{datasets}/{Type}/client{i}",device=device,model = model,
                        optim=args.optimizer,
                        backdoor_rate=1.0,
                        )
        scores[f'client{i}'] = 0
        client.setup(transform=None,batchsize=batchsize,local_epoch=local_epoch,
                     lr=args.lr)
        clients.append(client)
    have_create_client_num+=args.backdoor_num
    
    # =====
    # print(scores)

    alpha_init = 0.8
    beta_init = 0.2

    ''' resume '''
    if args.wandb_resume:
        max = -1
        folder_path = f'log/{exp_name}/score/{Type}/'
        for root,dirs,files in os.walk(folder_path):
            for file in files:
                if file.endswith('global.score.npy'):
                    file_path = os.path.join(root,file)
                    parent_dir = os.path.basename(os.path.dirname(file_path))
                    a = int(parent_dir)
                    if a > max:
                        max = a
    
    for epoch in range(max+1,args.epoch_num):
        ckpt_dir = f'log/{exp_name}/ckpt/{Type}/{epoch}/' 
        score_dir = f'log/{exp_name}/score/{Type}/{epoch}'
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        if not os.path.exists(score_dir):
            os.makedirs(score_dir)
        # model path
        globalmodel_path_pre = f'log/{exp_name}/ckpt/{Type}/{epoch-1}/global.ckpt'
        globalmodel_path = f'log/{exp_name}/ckpt/{Type}/{epoch}/global.ckpt'
        # score path 
        globalscore_path_pre = f'log/{exp_name}/score/{Type}/{epoch-1}/global.score.npy'
        globalscore_path = f'log/{exp_name}/score/{Type}/{epoch}/global.score.npy'
        
        if epoch==0:
            if not os.path.exists(f'log/{exp_name}/ckpt/{Type}/{epoch-1}/'):
                os.makedirs(f'log/{exp_name}/ckpt/{Type}/{epoch-1}')
            if not os.path.exists(f'log/{exp_name}/score/{Type}/{epoch-1}/'):
                os.makedirs(f'log/{exp_name}/score/{Type}/{epoch-1}')

            server.save_model(globalmodel_path_pre)
            np.save(globalscore_path_pre,scores)
        if epoch == max+1:
            server.get_globalmodel(globalmodel_path_pre)
            testloss,testacc = server.evaluate()
            logging.info(f'testloss={testloss}\ntestacc={testacc}')
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
            scores = client.caculate_scores(ckpt_dir,train_ids,
                                            alpha = alpha_init+(beta_init-alpha_init)*(epoch/args.epoch_num),
                                            beta = beta_init-(beta_init-alpha_init)*(epoch/args.epoch_num))
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
    
    if args.wandb_log:
        wandb.save(f'log/{exp_name}/log')
        run.finish()