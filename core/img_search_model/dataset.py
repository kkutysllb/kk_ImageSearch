#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-21 18:27
# @Desc   : 构建数据集
# --------------------------------------------------------
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader





class Dataset:
    def __init__(self, root_dir, batch_size, shuffle, num_workers, transform, is_train) -> None:
        self.root_dir = root_dir
        # 数据集假造
        classes, class_to_idx = datasets.folder.find_classes(self.root_dir)
        image_paths = datasets.ImageFolder.make_dataset(
            directory=self.root_dir,
            class_to_idx=class_to_idx,
            extensions=('jpg', 'jpeg', 'png')
        )
        self.image_paths = [img[0] for img in image_paths]
        if transform is None:
            transform = self.transforms_img(is_train=is_train)
        # 基于torchvision里面的API构建遍历对象
        self.dataset = datasets.ImageFolder(
            root=self.root_dir,
            transform=transform
        )
        # prefetch_factor = 2 if num_workers == 0 else num_workers * batch_size
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            # prefetch_factor=prefetch_factor,
            collate_fn=None
        )
    
    
    def __len__(self):
        return len(self.dataset.imgs)
    
    def __iter__(self):
        for img in self.loader:
            yield img
            
    def transforms_img(slef, is_train=is_train):
        if is_train:
        return transforms.Compose([
            # transforms.Lambda(lambda x: x.convert('RGB')), # 转换为RGB格式
            transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)), # 随机裁剪
            transforms.RandomRotation(degrees=15), # 随机旋转
            transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
            transforms.Resize(size=(224, 224)), # 调整图像大小
            transforms.ToTensor(), # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化
        ])
    else:
        return transforms.Compose([
            # transforms.Lambda(lambda x: x.convert('RGB')), # 转换为RGB格式
            transforms.Resize(size=(224, 224)), # 调整图像大小
            transforms.ToTensor(), # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化
        ])
            
    @staticmethod
    def convert_img(img):
        return img.convert('RGB')
    
    @staticmethod
    def get_online_transforms():
        return transforms.Compose([
            transforms.Lambda(lambd=self.convert_img),
            transforms.ToTensor(), # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化
        ])


if __name__ == "__main__":
    pass
