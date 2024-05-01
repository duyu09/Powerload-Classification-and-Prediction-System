# __encoding: utf-8 __
"""
@Version  : 1.0.0
@Time     : 2024年04月23日 (最近修改：2024年04月25日)
@Author   : 杜宇 (DuYu, @Duyu09, qluduyu09@163.com)、李庆隆
@File     : app-ai-mq.py
@Desc     : 使用RabbitMQ的AI算力端电力数据处理程序。
@Copyright: Copyright © 2024 The research and development group for
            Power Load Classification and Prediction System Based on Deep Learning Algorithm,
            Faculty of Computer Science & Technology, Qilu University of Technology (Shandong Academy of Sciences).
            齐鲁工业大学（山东省科学院）计算机科学与技术学部 “基于深度学习算法的电力负载分类与预测系统项目”研究与开发小组 保留所有权利。
@Note     : 暂无全局文件说明。
"""

import torch
import keras
import pika
import ast
import pickle
import numpy as np
from keras.models import load_model
from sklearn.decomposition import PCA
from prediction.prediction import *

classification_model_path = r"./classify/Classification-Model-v6.3-2.2.keras"
label2index_dict_path = r"./classify/label2index_dict.pkl"
pca_pkl_path = r"./classify/pca.pkl"
prediction_large_model_path = r"./prediction/model-large.pt"


def classify_model_usage(data: np.ndarray) -> str:
    """
    分类模型的使用
    :param data: 一维的ndarray数组数据
    :return: 表示电器的文本标签（如：“热水器”）
    """
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
        with open(pca_pkl_path, 'rb') as f:  # 读取pca对象，将训练时的参数用于推理时的特征处理。
            pca = pickle.load(f)
        feaArr = pca.transform(np.array(feaArr))
        return feaArr

    feaArr = getFeature(data)
    model = load_model(classification_model_path)  # 加载模型
    index = np.argmax(model.predict(feaArr)[0])
    label = ""
    with open(label2index_dict_path, 'rb') as f:
        l2i_dict = dict(pickle.load(f))  # 加载文本标签与数字编码对应关系字典
    for i in l2i_dict.items():
        if index == i[1]:
            label = i[0]
            break
    return label


# 预测模型的使用
# 输入数据和文本标签，输出预测数据
def prediction_model_usage(data: np.ndarray, label: str, max_length=200) -> np.ndarray:
    """
    预测模型的使用
    :param data: 一维的ndarray数组数据
    :param label: 文本标签（“如：热水器”）
    :param max_length: 输入到模型的最大数据长度（相当于对data截取后面的部分）
    :return: 后20时间步的预测数据
    """
    model = torch.load(prediction_large_model_path)  # 加载预测模型
    data = np.array(data)
    if data.shape[0] > max_length:
        data = data[:max_length]
    pred = usage(model, data, label)
    return pred


# 测试函数功能
def function_test():
    """
    对分类模型与预测模型的函数测试。
    :return: None
    """
    data = np.loadtxt('./dataset/电脑+热水壶.csv', delimiter=',', usecols=4, skiprows=1)[:800]
    label = classify_model_usage(data)
    print(prediction_model_usage(data, label))


def callback(ch, method, properties, body):
    """
    RabbitMQ的回调函数，模型推理在此函数中
    """
    # 此处进行模型推理计算
    result_dict = ast.literal_eval(body.decode('utf-8'))
    data = np.array(result_dict['data'])
    label = classify_model_usage(data)
    pred = prediction_model_usage(data, label)
    pred = map(int, pred)
    result = {"label": label, "pred": list(pred)}

    # 返回结果给后端服务器
    ch.basic_publish(exchange='', routing_key=properties.reply_to,
                     properties=pika.BasicProperties(correlation_id=properties.correlation_id),
                     body=str(result).encode('utf-8'))  # 将数据按照UTF-8字符串编码
    ch.basic_ack(delivery_tag=method.delivery_tag)


if __name__ == '__main__':
    # 连接到RabbitMQ服务器
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='task_queue', durable=True)  # 声明一个队列
    channel.basic_qos(prefetch_count=1)  # 告诉RabbitMQ，一旦消费者处理了任务，就可以从队列中删除它
    channel.basic_consume(queue='task_queue', on_message_callback=callback)
    print('app-ai-mq has started. Waiting for messages.')
    channel.start_consuming()
