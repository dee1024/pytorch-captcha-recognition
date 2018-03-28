# -*- coding: UTF-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dsets
import mydataset
import onehotencoding as ohe

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4
char_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
CHAR_SET_LEN = len(char_set)

transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
])

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear(7*20*64, 1024),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU())
        self.rfc = nn.Sequential(
            nn.Linear(1024, MAX_CAPTCHA*CHAR_SET_LEN),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.rfc(out)
        return out

def main():
    cnn = CNN()
    cnn.load_state_dict(torch.load('model.pkl'))
    print("load cnn net.")

    pre_dataset = dsets.ImageFolder(
        root='dataset/predict',
        transform=transform1
    )

    pre_loader = torch.utils.data.DataLoader(dataset=pre_dataset,batch_size=1,shuffle=False)

    for images in pre_loader:
        images = Variable(images[0])
        predict_label = cnn(images)
        c0 = np.argmax(predict_label[0, 0:10].data.numpy())
        c1 = np.argmax(predict_label[0, 10:20].data.numpy())
        c2 = np.argmax(predict_label[0, 20:30].data.numpy())
        c3 = np.argmax(predict_label[0, 30:40].data.numpy())
        c = '%s%s%s%s' % (c0, c1, c2, c3)
        print(c)

if __name__ == '__main__':
    main()


