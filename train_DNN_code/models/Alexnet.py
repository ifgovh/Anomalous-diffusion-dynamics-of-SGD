# Identical copies of two AlexNet models
import torch
import torch.nn as nn
import copy 
import torch.nn.functional as F
import torch.optim as optim

class FullyConnected(nn.Module):

    def __init__(self, input_dim=28*28 , width=50, depth=3, num_classes=10):
        super(FullyConnected, self).__init__()
        self.input_dim = input_dim 
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        
        layers = self.get_layers()

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.width, bias=False),
            nn.ReLU(inplace=True),
            *layers,
            nn.Linear(self.width, self.num_classes, bias=False),
        )

    def get_layers(self):
        layers = []
        for i in range(self.depth - 2):
            layers.append(nn.Linear(self.width, self.width, bias=False))
            layers.append(nn.ReLU())
        return layers

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = self.fc(x)
        return x


# This is a copy from online repositories 
class AlexNet(nn.Module):

    def __init__(self, input_height=32, input_width=32, input_channels=3, ch=64, num_classes=1000):
        # ch is the scale factor for number of channels
        super(AlexNet, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

        self.features = nn.Sequential(
            nn.Conv2d(3, out_channels=ch, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.size = self.get_size()
        print(self.size)
        a = torch.tensor(self.size).float()
        b = torch.tensor(2).float()
        self.width = int(a) * int(1 + torch.log(a) / torch.log(b))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.size, self.width),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.width, self.width),
            nn.ReLU(inplace=True),
            nn.Linear(self.width, num_classes),
        )

    def get_size(self):
        # hack to get the size for the FC layer...
        x = torch.randn(1, self.input_channels, self.input_height, self.input_width)
        y = self.features(x)
        print(y.size())
        return y.view(-1).size(0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def alexnet():
    return AlexNet(ch=64, num_classes=10)


def fc3(**kwargs):
    return FullyConnected(input_dim=32*32*3, width=100, depth=3, num_classes=10)

def fc20(**kwargs):
    return FullyConnected(input_dim=32*32*3, width=100, depth=20, num_classes=10)

def fc56(**kwargs):
    return FullyConnected(input_dim=32*32*3, width=100, depth=56, num_classes=10)

def fc110(**kwargs):
    return FullyConnected(input_dim=32*32*3, width=100, depth=110, num_classes=10)

def simplenet(**kwargs):
    return SimpleNet(**kwargs)
