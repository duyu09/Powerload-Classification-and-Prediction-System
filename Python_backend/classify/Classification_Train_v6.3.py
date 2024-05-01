# __encoding: utf-8 __
"""
@Version: 2.0.0
@Time:    2024年03月06日 (最近修改：2024年04月23日)
@Author:  杜宇 (DuYu, @Duyu09, qluduyu09@163.com)
@File:    Classification_Train_v6.3.py
@Desc:    电力负载分类模型训练代码。
@Note:    暂无全局文件说明。
"""

import os
import csv
import pickle
import sys
import numpy as np
from tqdm import tqdm
from keras import regularizers
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from collections import OrderedDict
from scipy.io import savemat, loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 配置全局变量与超参数
dc = r'../dataset'  # 读取该目录下的所有CSV文件
model_save_path = r'Classification-Model-v6.3-2.2.keras'  # 保存模型文件路径
matrix_save_path = r'Classification_feature-2.2.mat'  # 特征矩阵文件路径
dict_save_path = r"label2index_dict.pkl"  # 保存标签与数字编码对应的字典
pca_save_path = r"pca.pkl"  # 保存PCA对象
testRate = 0.15  # 测试集占数据集的比例 0.15
frameLength = 800  # 帧长度 800
step = 800  # 统计特征步长 800
max_value = 3000000  # 功率最大值 3000000
eps = 75  # 神经网络训练迭代次数 75
lamb = 0.001  # L1正则化惩罚系数 0.001
pca_n_components = 67  # PCA保留方差的比例(信息保留程度) 0.96
es_patience = 3  # 早停阈值(检测损失函数) 3
power_column = -1  # 功率值在数据集中的字段数(-1为倒数第1个字段)


# 读取directory目录下的CSV文件
def get_csv_files(directory):
    files = os.listdir(directory)
    csv_files_list = [file_name for file_name in files if file_name.endswith('.csv')]
    return csv_files_list


# 从文件加载数据并作简单处理
def load_data(csv_files, directory, have_head=True, power_column=-1):  # 倒数第1列是功率字段
    ret_data = OrderedDict()
    for csv_file in tqdm(csv_files, file=sys.stdout):
        with open(os.path.join(directory, csv_file), 'r') as file:
            csv_reader = csv.reader(file)
            if have_head:
                next(csv_reader)
            last_column = []
            for row in csv_reader:
                last_column.append(float(row[power_column]))
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
        with open(pca_save_path, 'wb') as f:
            pickle.dump(pca, f)
        print("PCA对象文件已保存到：", os.path.abspath(pca_save_path))
        original_labelArr = labelArr
        le = LabelEncoder()
        labelArr = le.fit_transform(np.array(labelArr))
        labelArr = to_categorical(labelArr)
        savemat(matrix_path, {'feature': feaArr, 'label': labelArr})  # 存储特征矩阵到文件
        print("特征矩阵文件已保存到：", os.path.abspath(matrix_path))
        label_dict = {}
        for i, label in enumerate(original_labelArr):
            if label.endswith(".csv"):
                label = label[:-4]
            label_dict[label] = np.where(labelArr[i] == 1)[0]
        with open(dict_save_path, 'wb') as f:
            pickle.dump(label_dict, f)
        print("字典文件已保存到：", os.path.abspath(dict_save_path))
    return feaArr, labelArr


# 主函数
def main():
    print("\n[正在读取文件]")
    csv_files = get_csv_files(dc)
    r_data = load_data(csv_files, dc, power_column=power_column)

    print("\n[正在进行特征提取]")
    rf, rl = getFeature(r_data, frame_length=frameLength, step=step, max_value=max_value, matrix_path=matrix_save_path)
    x_train, x_test, y_train, y_test = train_test_split(rf, rl, test_size=testRate)

    print("\n[正在构建神经网络结构]")
    len_csv_files = len(csv_files)
    model = Sequential()
    model.add(Dense(units=int(len_csv_files*4), activation='relu'))  # 394
    model.add(Dense(units=int(len_csv_files*2), activation='relu', kernel_regularizer=regularizers.l1(lamb)))  # 158
    model.add(Dense(units=int(len_csv_files), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='loss', patience=es_patience, restore_best_weights=True)

    print("\n[开始训练]")
    model.fit(x_train, y_train, epochs=eps, batch_size=28, verbose=1, callbacks=[early_stopping])

    print("\n[正在测试模型]")
    score = model.evaluate(x_test, y_test, batch_size=28)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print("\n[正在保存模型权重文件]")
    # save_path = os.path.join(dc, 'PLDA-Module-v6.3.keras')
    model.save(model_save_path)
    print("模型权重文件已保存到：", os.path.abspath(model_save_path))


if __name__ == '__main__':
    main()
