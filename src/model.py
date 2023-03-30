import torch.nn as nn
import torch
# Neurons of each layer
input_size = 784
hidden_size = 500  
num_classes = 10

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        x = x.reshape(-1, 28*28)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
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