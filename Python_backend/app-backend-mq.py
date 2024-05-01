# __encoding: utf-8 __
"""
@Version  : 1.0.0
@Time     : 2024年04月23日 (最近修改：2024年04月25日)
@Author   : 杜宇 (DuYu, @Duyu09, qluduyu09@163.com)、李庆隆
@File     : app-backend-mq.py
@Desc     : 使用RabbitMQ的分布式后端电力数据处理程序。
@Copyright: Copyright © 2024 The research and development group for
            Power Load Classification and Prediction System Based on Deep Learning Algorithm,
            Faculty of Computer Science & Technology, Qilu University of Technology (Shandong Academy of Sciences).
            齐鲁工业大学（山东省科学院）计算机科学与技术学部 “基于深度学习算法的电力负载分类与预测系统项目”研究与开发小组 保留所有权利。
@Note     : 暂无全局文件说明。
"""

import ast
import time
import uuid
import pika
import sqlite3
import numpy as np

conn = sqlite3.connect('powerload.db', check_same_thread=False)  # 创建数据库连接


def executeSQL(connect: sqlite3.Connection, sql: str, many=False, data=None) -> list:
    """
    执行SQL语句并返回执行结果集。
    :param connect: 连接对象
    :param sql: 将被执行的SQL语句
    :param many: 是否使用批量处理（是=True, 否=False）
    :param data: 使用批量处理时传入的批量数据
    :return: SQL语句的执行结果集
    """
    cursor = connect.cursor()
    if not many:
        cursor.execute(sql)
    else:
        cursor.executemany(sql, data)
    result = cursor.fetchall()
    conn.commit()
    cursor.close()
    return result


def get_realtime_data(user_id: int) -> np.ndarray:
    """
    读取指定user_id的实时监测数据：读取后800个数据，若不够800个，则进行重复，凑足800个数据。
    :param user_id: 用户ID
    :return: 读到的实时数据。（一维ndarray数组）
    """
    real_time = []
    sql = "SELECT time_unixstamp, powerload_value FROM realtime_data WHERE user_id = " + str(
        user_id) + " ORDER BY time_unixstamp DESC LIMIT 800"
    result = executeSQL(conn, sql)
    result.sort(key=lambda x: x[0])
    result_len = len(result)
    if result_len == 0:
        real_time = [0] * 800
    elif result_len >= 800:  # 没有特殊情况，不会出现>800个结果。
        real_time = [x[1] for x in result][:800]
    else:
        real_time = ([x[1] for x in result] * int(800 // result_len + 1))[:800]
    return np.array(real_time)


def get_userid_list() -> list:
    """
    从数据库读取user_id的列表，返回包含所有user_id整数的list。
    :return: 包含所有user_id整数的列表
    """
    sql = "SELECT DISTINCT user_id FROM user"
    result = executeSQL(conn, sql)
    result = [x[0] for x in result]
    return result


def write_database(user_id: int, label: str, pred_value: np.ndarray):
    """
    将请求结果写入数据库。
    :param user_id: 用户ID
    :param label: 分类的文本标签（如：“热水器”）
    :param pred_value: 预测的功率数值ndarray数组
    :return: None
    """
    unix_time = int(time.time())
    insert_label_sql = "INSERT INTO label_data (label_id, user_id, time_unixstamp, label_value) VALUES ((SELECT MAX(label_id) + 1 FROM label_data), " + \
                       ", ".join([str(user_id), str(unix_time), '"{}"'.format(label)]) + ");"
    executeSQL(conn, insert_label_sql)
    result = executeSQL(conn, "SELECT MAX(pred_id) FROM prediction_data")[0][0]
    pred_len = pred_value.shape[0]
    pred_id_arr = np.arange(result + 1, result + 1 + pred_len).reshape(-1, 1)
    user_id_arr = np.full(pred_len, user_id).reshape(-1, 1)
    time_unixstamp_arr = np.arange(unix_time, unix_time + pred_len).reshape(-1, 1)
    pred_value_arr = pred_value.reshape(-1, 1)
    arr = np.concatenate((pred_id_arr, user_id_arr, time_unixstamp_arr, pred_value_arr), axis=1)
    insert_prediction_sql = "INSERT INTO prediction_data (pred_id, user_id, time_unixstamp, pred_value) VALUES (?, ?, ?, ?);"
    executeSQL(conn, insert_prediction_sql, many=True, data=arr)


# 创建RPC客户端类，以便于处理MQ相关业务。
class RpcClient(object):
    def __init__(self):
        # 这里localhost需要改成AI算力端(app-ai-mq.py)所在地址，以连接MQ。
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, data):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='task_queue',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=str(data).encode("utf-8"))  # 将对象编码为UTF-8二进制字符串，然后发送。

        while self.response is None:
            self.connection.process_data_events()
        return self.response


def cir_calc():
    """
    反复向算力端发送数据，利用模型计算预测值与标签。\n
    请求数据格式约定：字典，键："data"，值：数组，该字典以UTF-8编码的二进制字符串的形式发送 \n
    响应数据格式约定：字典，键："label"，值：标签文本；键："pred"，值：预测数组，该字典以UTF-8编码的二进制字符串的形式发送
    :return: None
    """
    userid_list = get_userid_list()
    rpc_client = RpcClient()
    while True:
        for user_id in userid_list:
            realtime_data = list(get_realtime_data(user_id))
            response = rpc_client.call({"data": realtime_data}).decode('utf-8')
            response = ast.literal_eval(response)
            label, pred_values = response["label"], response["pred"]
            # write_database(user_id, str(label), np.array(pred_values))  # 写数据库
            print(pred_values)
            time.sleep(1)  # 隔1秒再处理下一个用户
        time.sleep(30)  # 隔30秒循环再次请求


if __name__ == '__main__':
    cir_calc()  # 开始执行循环请求。
