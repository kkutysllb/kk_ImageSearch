#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-21 19:32
# @Desc   : 自定义工具函数
# --------------------------------------------------------
"""

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __str__(self):
        return ' '.join(map(str, self.data))
    

def accuracy(y_hat, y):
    """定义分类精度"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
    
def convert_seconds(seconds):
    """秒数转换为时分秒格式"""
    hours, remainder = divmod(seconds, 3600)  # 1小时=3600秒
    minutes, seconds = divmod(remainder, 60)  # 1分钟=60秒
    return hours, minutes, seconds



if __name__ == "__main__":
    pass
