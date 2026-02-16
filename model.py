import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ImprovedCNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ImprovedCNN, self).__init__()
        
        #Block1: input3 -> output 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # fix1 BatchNorm
        
        #block2 : input 32 -> output 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) 
        
        #block3 : input 64 -> output 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) 
        
        #pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #dropout layer
        self.dropout = nn.Dropout(0.5) # fix2 Dropout
        
        #fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        #block1
        x = self.conv1(x)
        x = self.bn1(x) #bn before relu
        x = F.relu(x)
        x = self.pool(x)
        
        #block2 
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        #block3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        #flatten
        x = x.view(-1, 128 * 4 * 4)
        
        #classifier
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.dropout(x) #dropout before final layer
        
        x = self.fc2(x)
        return x
    