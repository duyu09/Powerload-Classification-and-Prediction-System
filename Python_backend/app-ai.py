# __encoding: utf-8 __
"""
@Version  : 1.0.0
@Time     : 2024年04月23日 (最近修改：2024年04月25日)
@Author   : 杜宇 (DuYu, @Duyu09, qluduyu09@163.com)
@File     : app-ai.py
@Desc     : 使用Flask的AI算力端电力数据处理程序。（本程序没有使用MQ，为另一种等效的解决方案）
@Copyright: Copyright © 2024 The research and development group for
            Power Load Classification and Prediction System Based on Deep Learning Algorithm,
            Faculty of Computer Science & Technology, Qilu University of Technology (Shandong Academy of Sciences).
            齐鲁工业大学（山东省科学院）计算机科学与技术学部 “基于深度学习算法的电力负载分类与预测系统项目”研究与开发小组 保留所有权利。
@Note     : 暂无全局文件说明。
"""

import sys
import torch
import keras
import pickle
import numpy as np
from keras.models import load_model
from sklearn.decomposition import PCA
from flask import Flask, request, jsonify
from prediction.prediction import *

app = Flask(__name__)
classification_model_path = r"./classify/Classification-Model-v6.3-2.2.keras"
label2index_dict_path = r"./classify/label2index_dict.pkl"
pca_pkl_path = r"./classify/pca.pkl"
prediction_large_model_path = r"./prediction/model-large.pt"


# 分类模型的使用
# 输入数据，输出文本类别
def classify_model_usage(data):
    def rate(arr, n_max, n_min):
        """
        计算比例 (此函数要重点优化执行效率)。
        """
        mask = (arr > n_min) & (arr < n_max)
        return np.mean(mask)

    def getFeature(data, frame_length=800, step=800, max_value=3000000):
        """
        计算data的特征向量。
        """
        data = data[:frame_length]
        feaArr = [[rate(data, i + step + 1, i) for i in range(step, max_value, step)]]
        with open(pca_pkl_path, 'rb') as f:  # 读取pca对象，将训练时的参数用于推理。
            pca = pickle.load(f)
        feaArr = pca.transform(np.array(feaArr))
        return feaArr

    feaArr = getFeature(data)
    model = load_model(classification_model_path)  # 加载模型
    index = np.argmax(model.predict(feaArr)[0])
    label = ""
    with open(label2index_dict_path, 'rb') as f:
        l2i_dict = dict(pickle.load(f))
    for i in l2i_dict.items():
        if index == i[1]:
            label = i[0]
            break
    return label


# 预测模型的使用
# 输入数据和文本标签，输出预测数据
def prediction_model_usage(data, label: str, max_length=200):
    model = torch.load(prediction_large_model_path)
    data = np.array(data)
    if data.shape[0] > max_length:
        data = data[:max_length]
    pred = usage(model, data, label)
    return pred


# 测试函数功能
def function_test():
    data = np.loadtxt('./dataset/电脑+热水壶.csv', delimiter=',', usecols=4, skiprows=1)[:800]
    label = classify_model_usage(data)
    print(prediction_model_usage(data, label))


@app.route("/model-working", methods=["POST"])
def model_working():
    data = request.json.get("data")
    data = np.array(data)
    label = classify_model_usage(data)
    pred = prediction_model_usage(data, label)
    pred = map(int, pred)
    return jsonify({"label": label, "pred": list(pred)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
