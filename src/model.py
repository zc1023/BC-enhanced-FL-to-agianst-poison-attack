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

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 14 * 14, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = x.view(-1, 16 * 14 * 14)
        x = F.relu(self.bn2(self.fc1(x)))
        x = self.fc2(x)
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

# class MLPLoan(nn.Module):
#     def __init__(self):
#         super(MLPLoan,self).__init__()
#         self.norm = nn.BatchNorm1d(195)
#         self.fc1 = nn.Linear(195, 320)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(320, 160)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(160, 80)
#         self.relu3 = nn.ReLU()
#         self.fc4 = nn.Linear(80, 40)
#         self.relu4 = nn.ReLU()
#         self.fc5 = nn.Linear(40, 20)
#         self.relu5 = nn.ReLU()
#         self.fc6 = nn.Linear(20, 2)
#     def forward(self, x):
#         x = self.norm(x)
#         out = self.fc1(x)
#         out = self.relu1(out)
#         out = self.fc2(out)
#         out = self.relu2(out)
#         out = self.fc3(out)
#         out = self.relu3(out)
#         out = self.fc4(out)
#         out = self.relu4(out)
#         out = self.fc5(out)
#         out = self.relu5(out)
#         out = self.fc6(out)
#         return out

class MLPLoan(nn.Module):
    def __init__(self):
        super(MLPLoan, self).__init__()
        self.fc1 = nn.Linear(195, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm1d(195)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
if __name__ == '__main__':
    pass
    # mlp = MNISTCNN()

    # num_epochs = 5
    # batch_size = 1024 # Recall that we set it before
    # learning_rate = 1e-3
    # criterion = nn.CrossEntropyLoss()  
    # optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

    # # from torchvision import datasets,transforms

    # # #导入训练数据
    # # train_dataset = datasets.MNIST(root='data/mnist',                #数据集保存路径
    # #                             train=True,                      #是否作为训练集
    # #                             transform=transforms.ToTensor(), #数据如何处理, 可以自己自定义
    # #                             download=True)                  #路径下没有的话, 可以下载
                                
    # # #导入测试数据
    # # test_dataset = datasets.MNIST(root='data/mnist',
    # #                             train=False,
    # #                             transform=transforms.ToTensor())
    # # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, #分批
    # #                                        batch_size=batch_size,
    # #                                        shuffle=True)          #随机分批

    # # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    # #                                       batch_size=batch_size,
    # #                                       shuffle=False)

    # from datasets import LocalDataset
    # from torch.utils.data import Dataset,DataLoader,random_split
    # from torchvision import transforms
    # data = LocalDataset(data_dir='data/MNIST/mnist_by_class',transform=transforms.Compose([transforms.ToTensor(),]))
    # train_loader = DataLoader(data,batch_size=batch_size,shuffle=True)
    # for epoch in range(num_epochs):
    #     mlp.train()
    #     for i, (images, labels) in enumerate(train_loader):
    #         outputs = mlp(images)
    #         loss = criterion(outputs, labels)
    #         optimizer.zero_grad()                          #清零梯度
    #         loss.backward()                                #反向求梯度
    #         optimizer.step()
            
    #         if (i+1) % 100 == 0:
    #             print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        
    #     mlp.eval()      #测试模式，关闭正则化
    #     correct = 0
    #     total = 0
    #     for images,labels in train_loader:
    #         outputs = mlp(images)
    #         _, predicted = torch.max(outputs, 1)   #返回值和索引
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #     print('测试准确率: {:.4f}'.format(100.0*correct/total))
    # from torchsummary import summary
    # model = Cifar10CNN().to('cuda')
    # summary(model,input_size=(3,32,32),batch_size=32)

    from datasets import TextDataset
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import StepLR
    dataset = TextDataset("/home/v-zhoucha/BC-enhanced-FL-to-agianst-poison-attack/data/kaggle_loans/2/backdoor.csv")
    dataloader = DataLoader(dataset=dataset, # 传入的数据集, 必须参数
                               batch_size=512,       # 输出的batch大小
                               shuffle=True,       # 数据是否打乱
                               num_workers=0)      # 进程数, 0表示只有主进程
    validdata = TextDataset("/home/v-zhoucha/BC-enhanced-FL-to-agianst-poison-attack/data/kaggle_loans/valid.csv")
    validdataloader = DataLoader(
        dataset=validdata,
        batch_size=512,       # 输出的batch大小
        shuffle=True,       # 数据是否打乱
        num_workers=0
    )
    model = MLPLoan().to("cuda")
    num_epochs = 50
    batch_size = 1024 # Recall that we set it before
    learning_rate = 0.01
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.9)

    model.eval()
    correct = 0
    for data in validdataloader:
        features = data[0].to("cuda")
        labels = data[1].type(torch.LongTensor).to("cuda")
        # input(features)
        # input(labels)
        output = model(features)
        _,output = torch.max(output, 1)
        # print(output.shape)
        correct += (output == labels).sum().item()
    print(correct/len(validdata))

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        for data in dataloader:
            # input(data)
            features = data[0].to("cuda")
            labels = data[1].type(torch.LongTensor).to("cuda")
            # input(features.shape)
            # input(labels.shape)
            output = model(features)
            
            loss = criterion(output,labels)
            _,output = torch.max(output, 1)
            # print(output.shape)
            correct += (output == labels).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            train_loss+=loss
        # scheduler.step()
        print(f"train_loss = {train_loss},train_acc={correct/len(dataset)}")
        model.eval()
        correct = 0
        for data in validdataloader:
            features = data[0].to("cuda")
            labels = data[1].type(torch.LongTensor).to("cuda")

            output = model(features)
            _,output = torch.max(output, 1)
            # print(output.shape)
            correct += (output == labels).sum().item()
        print(correct/len(validdata))
    