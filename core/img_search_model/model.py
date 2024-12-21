#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-21 18:57
# @Desc   : 图像检索模型
# --------------------------------------------------------
"""
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from core.img_search_model.dataset import Dataset, transforms_img
from core.img_search_model.utils import Accumulator, accuracy, convert_seconds   


class ImageSearchModel(object):
    def __init__(self, net, model_dir, logs_dir batch_size, num_workers, device, criterion, optimizer) -> None:
        self.model_dir = model_dir
        self.logs_dir = logs_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        self.net = net
        self.net.to(self.device)
        self.loss_fn = criterion
        self.opt = optimizer
        
        # 如果保存模型的目录不存在则直接创建
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 如果保存日志的目录不存在则直接创建
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.start_epoch = 0
        self.start_training_batch = 0
        
        # 恢复模型参数
        try:
            original = self.restore_model()
            self.opt.load_state_dict(original['opt'].state_dict())
            self.start_epoch = original.get('epoch', self.start_epoch)
            self.start_training_batch = original.get('train_batch', self.start_training_batch)
        except Exception as e:
            print(e)
       
    def training(self, train_data_dir, valid_data_dir, total_epoch, summary_batch_interval=2, save_epoch_interval=1):
        # 1. 加载数据
        trainset = Dataset(
            root_dir=train_data_dir,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            transform=transforms_img(is_train=True)
        )
        validset = Dataset(
            root_dir=valid_data_dir,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
            transform=transforms_img(is_train=False)
        )
        
        # 2. 记录输出
        writer = SummaryWriter(log_dir=self.logs_dir)
        
        # 3. 训练模型
        metrics = Accumulator(3)
        train_batch = self.start_training_batch
        total_epoch += self.start_epoch
        for epoch in range(self.start_epoch, total_epoch):
            print(f"Epoch:【{epoch + 1}/{total_epoch}】")
            self.net.train(True)
            train_loss = []
            train_acc = []
            valid_loss = []
            valid_acc = []
            for idx, (img, label) in enumerate(trainset):
                img = img.to(self.device)
                label = label.to(self.device)
                outputs = self.net(img)
                loss = self.loss_fn(outputs, label)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                with torch.no_grad():
                    metrics.add(loss * img.shape[0], accuracy(outputs, label), img.shape[0])
                # 每summary_batch_interval个批次进行一次损失、精度统计
                if idx % summary_batch_interval == 0 or idx == len(trainset) - 1:
                    train_loss.append(metrics[0] / metrics[2])
                    train_acc.append(metrics[1] / metrics[2])
                    # 验证集验证
                    val_loss, val_acc = self._evaluate(validset)
                    valid_loss.append(val_loss)
                    valid_acc.append(val_acc)
                    # 记录日志
                    writer.add_scalar('training_loss', train_loss[-1], train_batch)
                    writer.add_scalar('training_acc', train_acc[-1], train_batch)
                    writer.add_scalar('valid_loss', valid_loss[-1], train_batch)
                    writer.add_scalar('valid_acc', valid_acc[-1], train_batch)
                    # 打印记录
                    print(f'Iter {idx:<6} '
                          f'训练损失: {train_loss[-1]:<.4f}, '
                          f'训练精度: {train_acc[-1]:<5.3%}, '
                          f'验证损失: {valid_loss[-1]:<.4f}, '
                          f'验证精度: {valid_acc[-1]:<5.3%}, '
                          f'训练设备: {str(self.device)}')
                train_batch += 1
            
            # 记录总日志
            writer.add_scalars('epoch_loss', {'train': train_loss, 'valid': valid_loss}, epoch)
            writer.add_scalars('epoch_acc', {'train': train_acc, 'valid': valid_acc}, epoch)
            # 间隔性保存模型参数
            if epoch % save_epoch_interval == 0:
                self.save_model(epoch, train_batch)
                
        # 3. 保存最终模型参数
        self.save_model(epoch=total_epoch, train_batch=train_batch)
        writer.close()
    def save_model(self, epoch, train_batch):
        model_path = os.path.join(self.model_dir, f"model_{epoch:04d}.pth")
        torch.save(
            obj=
                {
                'net': self.net.state_dict(),
                'opt': self.opt.state_dict(),
                'epoch': epoch + 1,
                'train_batch': train_batch
                }, 
            f=model_path)
        
    def convert_jit_model(self):
        self.restore_model()
        m = torch.jit.script(self.net)
        torch.jit.save(m, os.path.join(os.path.dirname(self.model_dir), 'model.pt'))
        
    def restore_model(self):
        names = os.listdir(self.model_dir)
        if len(names) > 0:
            names.sort()
            name = names[-1]
            original = torch.load(os.path.join(self.model_dir, name))
            net_state_dict = original['net'].state_dict()
            self.net.load_state_dict(net_state_dict)
            missing_keys, unexpected_keys = self.net.load_state_dict(net_state_dict, strict=False)
            if len(missing_keys) > 0:
                print(f"当前模型参数:{missing_keys}未进行恢复/没有覆盖！！")
            self.opt.load_state_dict(original['opt'].state_dict())
            self.start_epoch = original['epoch']
            self.start_training_batch = original['train_batch']
        else:
            raise ValueError(f"当前模型目录:{self.model_dir}不存在模型参数！！")
        return original
    
    def _evaluate(self, validset):
        self.net.eval()
        metrics = Accumulator(3)
        with torch.no_grad():
            for img, label in validset:
                img = img.to(self.device)
                label = label.to(self.device)
                outputs = self.net(img)
                loss = self.loss_fn(outputs, label)
                metrics.add(loss * img.shape[0], accuracy(outputs, label), img.shape[0])
        val_loss = metrics[0] / metrics[2]
        val_acc = metrics[1] / metrics[2]
        return val_loss, val_acc
        

if __name__ == "__main__":
    pass
