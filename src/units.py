import torch
def model_params_to_matrix(model):
    # 初始化一个空的列表，用于存储模型参数
    params_list = []
    # 遍历模型的所有参数
    for param in model.parameters():
        # 将参数展平为一维，并添加到列表中
        params_list.append(param.view(-1))
    # 将列表中的所有参数堆叠成一个二维张量，并返回
    return torch.cat(params_list).to('cpu')