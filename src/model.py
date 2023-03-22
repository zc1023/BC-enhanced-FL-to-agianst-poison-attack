import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 定义三个全连接层
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)
        # 定义一个激活函数
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 将输入x展平为一维向量
        x = x.view(-1, 784)
        # 依次经过三个全连接层和激活函数
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x