#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-21 13:34
# @Desc   : 前段app
# --------------------------------------------------------
"""
import logging
from flask import Flask, request, jsonify
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from img_search.processer import Processer


app = Flask(__name__)
processer = Processer()


@app.route('/')
def index():
    return "简易图像搜索系统"


@app.route('/search', methods=['POST'])
def search():
    try:
        # 1. 获取参数
        img = request.form.get('image')
        if img is None:
            return jsonify({
                'code': 400,
                'msg': '必须给定图像base64转换后的image入参，当前不存在image参数。'
            })
        
        
        # 2. 逻辑代码处理(图像恢复、特征向量提取、相似图像检索)
        result = processer.process_image_search(img)
        return jsonify({
            'code': 200,
            'data': result
        })
    except Exception as e:
        logging.error("服务器异常。", exc_info=e)
        return jsonify({
            'code': 500,
            'msg': f'服务器执行异常，具体异常信息为:{e}'
        })


if __name__ == "__main__":
    pass
