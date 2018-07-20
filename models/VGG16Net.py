#coding:utf8
from .BasicModule import BasicModule
import torch.nn as nn
from torch.nn import functional as F


class VGG16(BasicModule):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            tnn.BatchNorm2d(64),
            tnn.ReLU(),

            # 1-2 conv layer
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 1 Pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(

            # 2-1 conv layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 2-2 conv layer
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 2 Pooling lyaer
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(

            # 3-1 conv layer
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 3-2 conv layer
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            #3-3 conv layer
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 3 Pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(

            # 4-1 conv layer
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # 4-2 conv layer
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            #4-3
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # 4 Pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer5 = nn.Sequential(

            # 5-1 conv layer
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # 5-2 conv layer
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            #5-3
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # 5 Pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer6 = nn.Sequential(

            # 6 Fully connected layer
            # Dropout layer omitted since batch normalization is used.
            nn.Linear(512*7*7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU())

        self.layer7 = nn.Sequential(

            # 7 Fully connected layer
            # Dropout layer omitted since batch normalization is used.
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU())

        self.layer8 = nn.Sequential(

            # 8 output layer
            nn.Linear(4096, 2),
            nn.BatchNorm1d(2),
            nn.Softmax())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        vgg16_features = out.view(out.size(0), -1)
        out = self.layer6(vgg16_features)
        out = self.layer7(out)
        out = self.layer8(out)


        return out
