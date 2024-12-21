#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-21 18:49
# @Desc   : 采用预训练的restnet50作为特征提取网络
# --------------------------------------------------------
"""
import torch.nn as nn
from torchvision import models


class Network(nn.Module):
    def __init__(self) -> None:
        super(Network, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        # 冻结参数
        for param in self.resnet50.parameters():
            param.requires_grad = False
        # 修改最后一层全连接层
        self.resnet50.fc = nn.Linear(in_features=2048, out_features=1024)
        self.resnet50.activation = nn.ReLU(inplace=True)
        self.resnet50.fc_extend = nn.Linear(in_features=1024, out_features=256)
        self.resnet50.fc_extend_activation = nn.ReLU(inplace=True)
        self.resnet50.fc_out = nn.Linear(in_features=256, out_features=2)

    def forward(self, img):
        z = self.resnet50(img)
        z = self.resnet50.activation(z)
        z = self.resnet50.fc_extend(z)
        z = self.resnet50.fc_extend_activation(z)
        z = self.resnet50.fc_out(z)
        return z


if __name__ == "__main__":
    model = Network()
    print(model)
