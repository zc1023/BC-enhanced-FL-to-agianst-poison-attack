import torch.nn as nn
import torch
# Neurons of each layer



class MLP(nn.Module):
    def __init__(self,dataset="cifar10"):
        self.data = dataset
        if self.data == "mnist":
            input_size = 28*28
            num_classes = 10
        elif self.data == "cifar10":
            input_size =32*32*3
            num_classes =10
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)  
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(64, num_classes)
    def forward(self, x):
        if self.data == "mnist":
            x = x.reshape(-1, 28*28)
        elif self.data == "cifar10":
            x = x.reshape(-1, 32*32*3)
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        return out

if __name__ == '__main__':

    mlp = MLP()

    num_epochs = 5
    batch_size = 1024 # Recall that we set it before
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate)

    # from torchvision import datasets,transforms

    # #导入训练数据
    # train_dataset = datasets.MNIST(root='data/mnist',                #数据集保存路径
    #                             train=True,                      #是否作为训练集
    #                             transform=transforms.ToTensor(), #数据如何处理, 可以自己自定义
    #                             download=True)                  #路径下没有的话, 可以下载
                                
    # #导入测试数据
    # test_dataset = datasets.MNIST(root='data/mnist',
    #                             train=False,
    #                             transform=transforms.ToTensor())
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, #分批
    #                                        batch_size=batch_size,
    #                                        shuffle=True)          #随机分批

    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                       batch_size=batch_size,
    #                                       shuffle=False)

    from datasets import LocalDataset
    from torch.utils.data import Dataset,DataLoader,random_split
    from torchvision import transforms
    data = LocalDataset(data_dir='data/mnist_by_class',transform=transforms.Compose([transforms.ToTensor(),]))
    train_loader = DataLoader(data,batch_size=batch_size,shuffle=True)
    for epoch in range(num_epochs):
        mlp.train()
        for i, (images, labels) in enumerate(train_loader):
            outputs = mlp(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()                          #清零梯度
            loss.backward()                                #反向求梯度
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        
        mlp.eval()      #测试模式，关闭正则化
        correct = 0
        total = 0
        for images,labels in train_loader:
            outputs = mlp(images)
            _, predicted = torch.max(outputs, 1)   #返回值和索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('测试准确率: {:.4f}'.format(100.0*correct/total))