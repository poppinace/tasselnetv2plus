# -*- coding: utf-8 -*-
"""
@author: hao lu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class Encoder(nn.Module):
    def __init__(self, arc='tasselnetv2plus'):
        super(Encoder, self).__init__()
        if arc == 'tasselnetv2':
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2), stride=2),
                nn.Conv2d(16, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2), stride=2),
                nn.Conv2d(32, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2), stride=2),
                nn.Conv2d(64, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
        elif arc == 'tasselnetv2plus':
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2), stride=2),
                nn.Conv2d(16, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2), stride=2),
                nn.Conv2d(32, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2), stride=2),
                nn.Conv2d(64, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.encoder(x)
        return x


class Counter(nn.Module):
    def __init__(self, arc='tasselnetv2plus', input_size=64, output_stride=8):
        super(Counter, self).__init__()
        k = int(input_size / 8)
        avg_pool_stride = int(output_stride / 8)

        if arc == 'tasselnetv2':
            self.counter = nn.Sequential(
                nn.Conv2d(128, 128, (k, k), bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, 1)
            )
        elif arc == 'tasselnetv2plus':
            self.counter = nn.Sequential(
                nn.AvgPool2d((k, k), stride=avg_pool_stride),
                nn.Conv2d(128, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, 1)
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.counter(x)
        return x


class Normalizer:
    @staticmethod
    def cpu_normalizer(x, imh, imw, insz, os):
        # CPU normalization
        bs = x.size()[0]
        normx = np.zeros((imh, imw))
        norm_vec = dense_sample2d(normx, insz, os).astype(np.float32)
        x = x.cpu().detach().numpy().reshape(bs, -1) * norm_vec
        return x
    
    @staticmethod
    def gpu_normalizer(x, imh, imw, insz, os):
        _, _, h, w = x.size()            
        accm = torch.cuda.FloatTensor(1, insz*insz, h*w).fill_(1)           
        accm = F.fold(accm, (imh, imw), kernel_size=insz, stride=os)
        accm = 1 / accm
        accm /= insz**2
        accm = F.unfold(accm, kernel_size=insz, stride=os).sum(1).view(1, 1, h, w)
        x *= accm
        return x.squeeze().cpu().detach().numpy()


class CountingModels(nn.Module):
    def __init__(self, arc='tasselnetv2plus', input_size=64, output_stride=8):
        super(CountingModels, self).__init__()
        self.input_size = input_size
        self.output_stride = output_stride

        self.encoder = Encoder(arc)
        self.counter = Counter(arc, input_size, output_stride)
        if arc == 'tasselnetv2':
            self.normalizer = Normalizer.cpu_normalizer
        elif arc == 'tasselnetv2plus':
            self.normalizer = Normalizer.gpu_normalizer
        
        self.weight_init()

    def forward(self, x, is_normalize=True):
        imh, imw = x.size()[2:]
        x = self.encoder(x)
        x = self.counter(x)
        if is_normalize:
            x = self.normalizer(x, imh, imw, self.input_size, self.output_stride)
        return x

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_uniform_(
                #         m.weight, 
                #         mode='fan_in', 
                #         nonlinearity='relu'
                #         )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":

    from time import time

    insz, os = 64, 8
    imH, imW = 1080, 1920
    net = CountingModels(arc='tasselnetv2plus', input_size=insz, output_stride=os).cuda()
    with torch.no_grad():
        net.eval()
        x = torch.randn(1, 3, imH, imW).cuda()
        y = net(x)
        print(y.shape)

    import numpy as np

    with torch.no_grad():
        frame_rate = np.zeros((100, 1))

        for i in range(100):
            x = torch.randn(1, 3, imH, imW).cuda()
            torch.cuda.synchronize()
            start = time()

            y = net(x)

            torch.cuda.synchronize()
            end = time()

            running_frame_rate = 1 * float(1 / (end - start))
            frame_rate[i] = running_frame_rate
        print(np.mean(frame_rate))