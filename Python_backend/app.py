# __encoding: utf-8 __
"""
@Version  : 2.0.0
@Time     : 2024年04月15日 (最近修改：2024年04月23日)
@Author   : 杜宇 (DuYu, @Duyu09, qluduyu09@163.com)
@File     : app.py
@Desc     : 电力负载分类与预测系统 客户端服务代码（本程序基于Flask框架，等价于系统的C++ WebServer模块）。
@Copyright: Copyright © 2024 The research and development group for
            Power Load Classification and Prediction System Based on Deep Learning Algorithm,
            Faculty of Computer Science & Technology, Qilu University of Technology (Shandong Academy of Sciences).
            齐鲁工业大学（山东省科学院）计算机科学与技术学部 “基于深度学习算法的电力负载分类与预测系统项目”研究与开发小组 保留所有权利。
@Note     : 暂无全局文件说明。
"""

import uuid
import sqlite3
from flask import Flask, request, jsonify

app = Flask(__name__)
conn = sqlite3.connect('powerload.db', check_same_thread=False)


def executeSQL(connect, sql):
    """
    执行SQL语句并返回执行结果。
    传入连接对象和SQL语句。
    """
    cursor = connect.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    conn.commit()
    cursor.close()
    return result


# 存储用户的token，相同用户每次登录都更新token，故只能保持一个设备的在线。
# 键为用户ID(整型)，值为token字符串。
token_dict = dict()


def generate_token() -> str:
    """ 生成不带横线的GUID字符串作为token """
    guid = uuid.uuid4()
    guid_str = str(guid)
    guid_str_without_hyphens = guid_str.replace('-', '')
    return guid_str_without_hyphens


def add_token(token: str, userid: int, t_dict=token_dict):
    """ 向token字典添加token或替换token """
    t_dict[userid] = token


def verify_token(token: str, userid: int, t_dict=token_dict):
    """ 验证token """
    real_token = t_dict.get(userid)
    if real_token == token:
        return True
    else:
        return False


@app.route("/api/test", methods=["GET"])
def app_test():
    """ 测试函数：测试FLASK是否可以正常运行 """
    return "Hello, Test!"


@app.route("/api/login", methods=["POST"])
def login():
    """ 01号接口：用户登录 """
    phone_number = request.json.get("phone_number")
    password_md5 = request.json.get("password")
    result = executeSQL(conn, 'SELECT user_id, user_password FROM user WHERE user_phone_number = "' + str(phone_number) + '";')
    if len(result) == 0:
        return jsonify({'code': 1, 'token': ''})
    user_id = result[0][0]
    real_password_md5 = result[0][1]
    if real_password_md5 != password_md5:
        return jsonify({'code': 1, 'token': ''})
    token = generate_token()
    add_token(token, user_id)
    return jsonify({'code': 0, 'token': token})


@app.route("/api/register", methods=["POST"])
def register():
    """ 02号接口：用户注册 """
    phone_number = request.json.get("phone_number")
    password_md5 = request.json.get("password")
    room = request.json.get("room")
    r_tmp01 = executeSQL(conn, 'SELECT user_id FROM user WHERE user_phone_number = "' + str(phone_number) + '";')
    if len(r_tmp01) > 0:
        return jsonify({'code': 1, 'description': "该手机号已经被注册"})
    sql = "INSERT INTO user (user_id, user_password, user_phone_number, user_powermeter_id, user_room) VALUES " + \
    "((SELECT MAX(user_id) + 1 FROM user), " + ", ".join(['"{}"'.format(password_md5), '"{}"'.format(phone_number), "0", '"{}"'.format(room)]) + ");"
    executeSQL(conn, sql)
    return jsonify({'code': 0, 'description': ""})


@app.route("/api/getdata", methods=["POST"])
def getdata():
    """ 03号接口：读取负载数据 """
    phone_number = request.json.get("phone_number")
    token = request.json.get("token")
    r_tmp02 = executeSQL(conn, 'SELECT user_id FROM user WHERE user_phone_number = "' + str(phone_number) + '";')
    if len(r_tmp02) == 0:
        return jsonify({'code': 1, 'description': "该手机号未注册"})
    user_id = r_tmp02[0][0]
    if not verify_token(token, int(user_id)):
        return jsonify({'code': 1, 'description': "令牌错误"})

    r_tmp03 = executeSQL(conn, "SELECT time_unixstamp, powerload_value FROM realtime_data WHERE user_id = " + str(user_id) + " ORDER BY time_unixstamp DESC LIMIT 50")
    r_tmp03.sort(key=lambda x: x[0])
    realtime_data = list(map(lambda x: x[1], r_tmp03))
    if len(realtime_data) < 50:
        realtime_data = [0] * (50 - len(realtime_data)) + realtime_data
    realtime_data = list(map(lambda x: x / 1000, realtime_data))

    r_tmp04 = executeSQL(conn, "SELECT time_unixstamp, pred_value FROM prediction_data WHERE user_id = " + str(user_id) + " ORDER BY time_unixstamp DESC LIMIT 20")
    r_tmp04.sort(key=lambda x: x[0])
    prediction_data = list(map(lambda x: x[1], r_tmp04))
    if len(prediction_data) < 20:
        prediction_data = prediction_data + [0] * (20 - len(prediction_data))
    prediction_data = list(map(lambda x: x / 1000, prediction_data))

    r_tmp05 = executeSQL(conn, "SELECT label_value FROM label_data WHERE user_id = " + str(user_id) + " ORDER BY time_unixstamp DESC LIMIT 1")
    if len(r_tmp05) == 0:
        label_data = "正在计算..."
    else:
        label_data = str(r_tmp05[0][0])

    return jsonify({'code': 0, 'description': "", 'real_time': realtime_data, 'prediction': prediction_data, 'label': label_data})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

