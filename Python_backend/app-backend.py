# __encoding: utf-8 __
"""
@Version  : 1.0.0
@Time     : 2024年04月23日 (最近修改：2024年04月25日)
@Author   : 杜宇 (DuYu, @Duyu09, qluduyu09@163.com)
@File     : app-backend-mq.py
@Desc     : 对接基于Flask的AI端的分布式后端电力数据处理程序。（本程序没有使用MQ，为另一种等效的解决方案）
@Copyright: Copyright © 2024 The research and development group for
            Power Load Classification and Prediction System Based on Deep Learning Algorithm,
            Faculty of Computer Science & Technology, Qilu University of Technology (Shandong Academy of Sciences).
            齐鲁工业大学（山东省科学院）计算机科学与技术学部 “基于深度学习算法的电力负载分类与预测系统项目”研究与开发小组 保留所有权利。
@Note     : 暂无全局文件说明。
"""

import json
import time
import sqlite3
import requests
import numpy as np

conn = sqlite3.connect('powerload.db', check_same_thread=False)


# 执行SQL语句并返回执行结果。
# 传入连接对象和SQL语句。
def executeSQL(connect: sqlite3.Connection, sql: str, many=False, data=None):
    cursor = connect.cursor()
    if not many:
        cursor.execute(sql)
    else:
        cursor.executemany(sql, data)
    result = cursor.fetchall()
    conn.commit()
    cursor.close()
    return result


# 读取指定user_id的实时数据：读取后800个，若不够800个，则进行重复，凑足800个数据。
# 输入user_id，输出ndarray格式的实时数据。
def get_realtime_data(user_id: int):
    real_time = []
    sql = "SELECT time_unixstamp, powerload_value FROM realtime_data WHERE user_id = " + str(user_id) + " ORDER BY time_unixstamp DESC LIMIT 800"
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


# 从数据库读取user_id的列表，返回list。
def get_userid_list():
    sql = "SELECT DISTINCT user_id FROM user"
    result = executeSQL(conn, sql)
    result = [x[0] for x in result]
    return result


# 将请求结果写入数据库
# 输入参数：user_id，文本标签，预测值ndarray数组。
def write_database(user_id: int, label: str, pred_value: np.ndarray):
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


# 反复向算力端发送数据，利用模型计算预测值与标签。
# 请求数据格式约定：JSON，键："data"，值：数组
# 响应数据格式约定：JSON，键："label"，值：标签文本；键："pred"，值：预测数组
def cir_calc(url):
    userid_list = get_userid_list()
    while True:
        for user_id in userid_list:
            realtime_data = list(get_realtime_data(user_id))
            send_data = json.dumps({"data": realtime_data})
            response = requests.post(url, json=send_data)  # 发送请求
            if response.headers.get('content-type') == 'application/json':
                json_response = response.json()
            else:
                print("出现错误。")
                continue
            label, pred_values = json_response["label"], json_response["pred"]
            write_database(user_id, str(label), np.array(pred_values))
            time.sleep(0.3)  # 隔0.3秒再处理下一个用户。


if __name__ == '__main__':
    url = r"/model-working"
    # get_realtime_data(1)
    # print(get_userid_list())
    # array = np.random.rand(20)
    # write_database(1, "usb风扇+电吹风", array)

