# -*- coding: utf-8 -*-
"""
@Version: 2.0.0
@Time:    2024年03月06日 (最近修改：2024年04月24日)
@Author:  杜宇 (DuYu, @Duyu09, qluduyu09@163.com)
@File:    Classification_Test_v6.3.py
@Desc:    电力负载分类模型推理测试与调试代码。
@Note:    暂无全局文件说明。
"""

import os
import csv
import sys
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from scipy.io import savemat, loadmat
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 配置全局变量与超参数
dc = r"../dataset"  # 读取该目录下的所有CSV文件
model_path = r'Classification-Model-v6.3-2.2.keras'  # 模型文件路径
matrix_save_path = r'Classification_feature-2.2.mat'  # 特征矩阵文件路径
testRate = 0.9  # 测试集占数据集的比例
frameLength = 800  # 帧长度
step = 800  # 统计特征步长
max_value = 3000000  # 功率最大值 3000000
pca_n_components = 0.96  # PCA保留方差的比例(信息保留程度) 0.96


# 读取directory目录下的CSV文件
def get_csv_files(directory):
    files = os.listdir(directory)
    csv_files = [file_name for file_name in files if file_name.endswith('.csv')]
    return csv_files


# 从文件加载数据并作简单处理
def load_data(csv_files, directory, have_head=True):
    ret_data = OrderedDict()
    for csv_file in csv_files:
        with open(os.path.join(directory, csv_file), 'r') as file:
            csv_reader = csv.reader(file)
            if have_head:
                next(csv_reader)
            last_column = []
            for row in csv_reader:
                last_column.append(float(row[-2]))  # 倒数第二列是功率字段
        ret_data[csv_file] = np.array(last_column)
    return ret_data


# 计算比例 (此函数要重点优化执行效率)。
def rate(arr, n_max, n_min):
    mask = (arr > n_min) & (arr < n_max)
    return np.mean(mask)


# 获取特征矩阵和对应的标签数组
def getFeature(data, frame_length=1700, step=12, max_value=3000000, use_matrix=True, matrix_path=r"./PLDA_feature.mat"):
    feaArr, labelArr = [], []
    if not os.path.exists(os.path.join(matrix_path)):
        use_matrix = False
    if use_matrix:
        print("使用", os.path.abspath(matrix_path), "中的特征数据，不再进行特征工程。")
        mat_data = loadmat(matrix_path)
        feaArr, labelArr = mat_data['feature'], mat_data['label']
    else:
        feaArr, labelArr = [], []
        n = 0
        for arr in data:
            n = n + 1
            print(str(n) + '/' + str(len(data)), str(round(100 * n / len(data), 2)) + ' %')
            lengthNum = frame_length * int(len(data[arr]) / frame_length)
            aqr = data[arr][:lengthNum]
            arr02 = aqr.reshape((-1, frame_length))
            for q in tqdm(arr02, file=sys.stdout):
                tempArr = [rate(q, i + step + 1, i) for i in range(step, max_value, step)]
                feaArr.append(tempArr)
                labelArr.append(arr)
        pca = PCA(n_components=pca_n_components)
        feaArr = pca.fit_transform(np.array(feaArr))
        le = LabelEncoder()
        labelArr = le.fit_transform(np.array(labelArr))
        labelArr = to_categorical(labelArr)
        savemat(matrix_path, {'feature': feaArr, 'label': labelArr})  # 存储特征矩阵到文件
        print("特征矩阵文件已保存到：", os.path.abspath(matrix_path))
    return feaArr, labelArr


# 只打印综合准确度score的百分比值。
def print_score_value(rf, rl):
    x_train, x_test, y_train, y_test = train_test_split(rf, rl, test_size=testRate)
    score = model.evaluate(x_test, y_test, batch_size=24, verbose=0)
    print('第', (pcs + 1), '次预测：', round(score[1] * 100, 3), '%')


# 打印出错的组合，便于修改特征工程与调参。
def print_error_tuple(rf, rl):
    x_train, x_test, y_train, y_test = train_test_split(rf, rl, test_size=testRate)
    e = model.predict(x_test)
    predicted_classes = np.argmax(e, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    vec = predicted_classes - true_classes
    print("预测向量: ", predicted_classes)
    print("实际向量: ", true_classes)
    w = []
    for i in range(len(predicted_classes)):
        i01 = predicted_classes[i]
        i02 = true_classes[i]
        if not i01 == i02:
            w.append((i01, i02))
    print("错误分类元组：", w)


if __name__ == '__main__':
    print("\n[正在读取文件]")
    csv_files = get_csv_files(dc)
    r_data = load_data(csv_files, dc)
    print("特征向量标签", list(r_data.keys()))  # 打印特征向量标签

    print("\n[加载模型文件]")
    model = load_model(model_path)

    print("\n[正在进行特征提取]")
    rf, rl = getFeature(r_data, frame_length=frameLength, step=step, max_value=max_value, matrix_path=matrix_save_path)

    for pcs in range(5):  # 测试5次
        print("\n[第", pcs + 1, "次测试]")
        # print_score_value(rf, rl)
        print_error_tuple(rf, rl)
