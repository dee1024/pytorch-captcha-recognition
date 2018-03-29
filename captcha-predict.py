# -*- coding: UTF-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dsets
from visdom import Visdom # pip install Visdom

vis = Visdom()
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
char_set = NUMBER
MAX_CAPTCHA = 4
CHAR_SET_LEN = len(char_set)

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
        transform=transforms.ToTensor()
    )

    pre_loader = torch.utils.data.DataLoader(dataset=pre_dataset,batch_size=1,shuffle=False)


    # print(char_set)

    for images in pre_loader:
        image = images[0]
        # print(image.shape)
        vimage = Variable(image)
        predict_label = cnn(vimage)
        # c0 = char_set[np.argmax(predict_label[0, 0:36].data.numpy())]
        # c1 = char_set[np.argmax(predict_label[0, 36:72].data.numpy())]
        # c2 = char_set[np.argmax(predict_label[0, 72:108].data.numpy())]
        # c3 = char_set[np.argmax(predict_label[0, 108:144].data.numpy())]

        c0 = char_set[np.argmax(predict_label[0, 0:10].data.numpy())]
        c1 = char_set[np.argmax(predict_label[0, 10:20].data.numpy())]
        c2 = char_set[np.argmax(predict_label[0, 20:30].data.numpy())]
        c3 = char_set[np.argmax(predict_label[0, 30:40].data.numpy())]

        c = '%s%s%s%s' % (c0, c1, c2, c3)
        vis.images(image, opts=dict(caption=c))

if __name__ == '__main__':
    main()


