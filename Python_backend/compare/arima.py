import csv
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

# 忽略警告信息
warnings.filterwarnings("ignore")


def predict_and_plot_arima(series, order=(3, 1, 25), steps=20, ne=np.array([])):
    """
    使用ARIMA模型进行预测并绘制折线图。

    参数:
    series -- 输入的时间序列数据
    order -- ARIMA模型的参数(order=(p, d, q))
    steps -- 预测的步数
    """
    # 将numpy数组转换为pandas Series
    if isinstance(series, np.ndarray):
        series = pd.Series(series)

    # 训练ARIMA模型
    model = ARIMA(series, order=order)
    model_fit = model.fit()

    # 预测后100步
    forecast = model_fit.forecast(steps=steps)

    # 绘制原始数据和预测数据
    plt.figure(figsize=(12, 6))
    plt.plot(np.concatenate((series, ne)), label='Test')
    plt.plot(np.arange(len(series), len(series) + steps), forecast, label='Pred', color='red')
    plt.title('ARIMA Prediction')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    return forecast


if __name__ == '__main__':
    # 示例数据集 CSV 文件名为 'c.csv'
    csv_file = 'c.csv'

    # 使用 Python 内置的 csv 模块读取 CSV 文件。
    time_series = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 跳过标题行
        for row in csv_reader:
            # 假设第二列是时间序列数据
            time_series.append(float(row[0]))

    # 将列表转换为 NumPy 数组，因为 pmdarima 需要 NumPy 数组或 pandas Series
    time_series = np.array(time_series)
    sc = StandardScaler()
    time_series = sc.fit_transform(time_series.reshape(-1, 1)).reshape(-1, )

    # 调用函数
    predict_and_plot_arima(time_series[:100], ne=time_series[100:])
