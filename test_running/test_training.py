#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-21 19:56
# @Desc   : 模型训练测试
# --------------------------------------------------------
"""
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim

from core.img_search_model.model import ImageSearchModel
from core.img_search_model.network import Network
from core.img_search_model.dataset import transforms_img


super_dir = '/Users/libing/kk_datasets/'

def test_training():
    net = Network()
    model = ImageSearchModel(
        net=net,
        model_dir = os.path.join(root_dir, 'weights', 'DogCatClassifier'),
        logs_dir = os.path.join(root_dir, 'logs', 'DogCatClassifier'),
        batch_size=128,
        num_workers=2,
        device='cpu',
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)
    )
    
    model.training(
        train_data_dir=os.path.join(super_dir, 'dogcat', 'train'),
        valid_data_dir=os.path.join(super_dir, 'dogcat', 'test'),
        total_epoch=50,
        summary_batch_interval=10,
        save_epoch_interval=5
    )


if __name__ == "__main__":
    test_training()
