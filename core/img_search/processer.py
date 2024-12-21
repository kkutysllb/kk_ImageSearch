#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-21 18:20
# @Desc   : 图像搜索逻辑处理器
# --------------------------------------------------------
"""
import base64
import io
import numpy as np
import os
import pickle
import torch
import torchvision.transforms as transforms
from sklearn.neighbors import KDTree
from PIL import Image


def decode(img, img_path):
    img = base64.b64decode(img) # 解码还原图像
    img = Image.open(io.BytesIO(img))
    img.show()
    with open(img_path, 'wb') as writer:
        writer.write(img)


class Processer:
    def __init__(self, model_dir):
        print("内部处理器构建完成!!")
        self.model = torch.jit.load(os.path.join(model_dir, 'model.pt'))
        self.model.eval()
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.idx_2_names = ['小猫', '小狗']
        self.image_paths = self.load(os.path.join(model_dir, 'image_paths.pkl'))
        self.features = np.asarray(self.load(os.path.join(model_dir, 'features.pkl'))).astype(np.float32)

        # 对特征向量进行 L2 归一化，使欧氏距离等价于余弦距离
        self.features = self.features / np.linalg.norm(self.features, axis=1)[:, np.newaxis]
        self.tree = KDTree(self.features)
        
    @staticmethod
    def load(path):
        with open(path, 'rb') as reader:
            return pickle.load(reader)  


    def process_image_search(self, img, k=3):
        """
        具体的图像检索逻辑
        :param img: base64编码后的图像字符串
        :param k: 希望获取的k个相似图像
        :return: 结果列表
        """
        # 1. 将图像恢复
        img = decode(img, './tmp.png')
        img = self.transforms(img)[None, ...]  # [1,3,H,W]
        
        # 2. 调用模型获得图像向量
        vectors, scores = self.model(img)
        scores = torch.softmax(scores, dim=1)
        pred_index = torch.argmax(scores, dim=1)
        index = pred_index[0].item()
        class_name = self.idx_2_names[index]
        
        # 3. 基于向量索引库进行相似向量的搜索
        neighbors_dist, neighbors_indexs = self.tree.query(vectors.detach().numpy(), k=100)
        
        sim = []
        cnt = 0
        for r in zip(neighbors_dist[0], neighbors_indexs[0]):
            if r[0] > 5:
                # 距离超过5.0的样本直接不考虑
                continue
            sim.append({
                'img': f'{self.image_paths[r[1]]}',
                'score': f'{r[0]}'
            })
            cnt += 1
        
        return {
            'cnt': cnt,
            'class': class_name,
            'pro': float(f'{scores[0][index].item():.5f}'),
            'sim': sim
        }

if __name__ == "__main__":
    pass
