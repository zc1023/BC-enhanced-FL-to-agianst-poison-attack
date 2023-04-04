import torch.nn as nn
import torch
# Neurons of each layer
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,dataset="mnist"):
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

class MNISTCNN(nn.Module):

    def __init__(self):
        super(MNISTCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

class Cifar10CNN(nn.Module):

    def __init__(self):
        super(Cifar10CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool1(x)

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool2(x)

        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.pool3(x)

        x = x.view(-1, 128 * 4 * 4)

        x = self.fc1(x)
        x = F.softmax(self.fc2(x))

        return x


if __name__ == '__main__':

    mlp = MNISTCNN()

    num_epochs = 5
    batch_size = 1024 # Recall that we set it before
    learning_rate = 1e-3
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

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
    data = LocalDataset(data_dir='data/MNIST/mnist_by_class',transform=transforms.Compose([transforms.ToTensor(),]))
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