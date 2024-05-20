# __encoding: utf-8 __
"""
@Version  : 2.1.0
@Time     : 2024年04月15日 (最近修改：2024年05月20日)
@Author   : 杜宇 (DuYu, @Duyu09, qluduyu09@163.com)
@File     : prediction.py
@Desc     : 电力负载预测模型的训练、测试与使用代码。
@Copyright: Copyright © 2024 The research and development group for
            Power Load Classification and Prediction System Based on Deep Learning Algorithm,
            Faculty of Computer Science & Technology, Qilu University of Technology (Shandong Academy of Sciences).
            齐鲁工业大学（山东省科学院）计算机科学与技术学部 “基于深度学习算法的电力负载分类与预测系统项目”研究与开发小组 保留所有权利。
@Note     : 暂无全局文件说明。
"""

import os
import sys
import torch
import numpy as np
from math import sqrt
from tqdm import tqdm
from torch import nn, optim
from matplotlib import pyplot as plt
from torch.cuda.amp import GradScaler, autocast

# 全局变量与超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置训练及推理设备
transformer_block_drop_prob = 0.015  # 设置Transformer Block类中神经元Dropout的比率。
transformer_forecaster_drop_prob = 0.015  # 设置Transformer Forecaster类中神经元Dropout的比率。


def printlog(string: object) -> None:
    """
    函数功能：打印日志。
    :param string: 打印日志的内容。可以是任何对象，最终转为字符串进行打印。
    :return: None
    """
    print("".join(["[INFO] ", str(string)]))


# 类：Transformer Block类，包含一个多头注意力层和全连接神经网络层，多个Block可以叠加起来供预测器使用。
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, drop_prob=transformer_block_drop_prob):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True, device=device)
        self.fc = nn.Sequential(nn.Linear(embed_size, 4 * embed_size, device=device),
                                nn.LeakyReLU(),
                                nn.Linear(4 * embed_size, embed_size, device=device))
        self.dropout = nn.Dropout(drop_prob)
        self.ln1 = nn.LayerNorm(embed_size, eps=1e-5, device=device)
        self.ln2 = nn.LayerNorm(embed_size, eps=1e-5,device=device)

    def forward(self, x):
        x = x.float()
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)
        fc_out = self.fc(x)
        x = x + self.dropout(fc_out)
        x = self.ln2(x)
        return x


def get_data(filename: str, skip_header=1, usecol=0) -> np.ndarray:
    """
    函数说明：从CSV文件中读取数据，将小于等于0的值去除后直接返回。\n
    返回值形状：data:(数据长度,)

    :param filename: CSV文件名，用于读取数据。
    :param skip_header: 表头的行数，程序将跳过这些行。
    :param usecol: 有效数据位于的列号。
    :return: (ndarray) 形状：(大约为数据长度,) 返回读到并处理后的全部数据数组。
    """
    data = np.genfromtxt(filename, delimiter=',', skip_header=skip_header, usecols=[usecol], encoding="gbk")
    data = np.nan_to_num(data, nan=0, posinf=np.nanmax(data), neginf=np.nanmin(data))
    data[data < 0] = 0
    nonzero_indices = np.nonzero(data)
    data = data[nonzero_indices]
    return data


def scale_normal(data: np.ndarray, a_min=0.0001) -> tuple[np.ndarray, float, float]:
    """
    函数说明：归一化数据，使其均值为0，标准差为1。注意当标准差为0时，函数将以一个
    趋近于0的极小值a_min代替0。\n
    返回值形状：data:(数据长度,)

    :param data: 输入的ndarray原始数据。
    :param a_min: 当标准差为0时，用以代替之的小数值。
    :return: (tuple[np.ndarray, float, float]) 返回值01：形状：(数据长度,) 将原数据归一化后的数据。类型为ndarray；
    返回值02：平均值，类型为浮点数；返回值03：标准差，类型为浮点数。
    """
    mean = np.mean(data)
    std = np.clip(np.std(data), a_min, None)
    data = (data - mean) / std
    return data, mean, std


def window_data(data: np.ndarray, window_size=10) -> np.ndarray:
    """
    将数据划分为固定长度的窗口，并返回包含这些窗口的二维数组。注意：此函数会去除前面的一些数据点以匹配窗口。\n
    返回值形状：data:(窗口数量, 窗口大小【window_size】)
    :param data: 待被分窗的一维ndarray数组
    :param window_size: 窗口长度
    :return: 分窗后的二维数组 形状：(窗口数量, 窗口大小【window_size】)
    """
    data_len = (data.shape[0] // window_size) * window_size
    data = data[-data_len:]
    data = data.reshape(-1, window_size)
    return data


def make_frame(feature_matrix: np.ndarray, frame_size=10):
    """
    分帧函数，每帧包含若干窗，feature_matrix为加窗后的二维数组，frame_size为每帧的窗数(帧长)。
    该函数会丢弃前面几个时间窗以匹配帧长。

    :param feature_matrix: 加窗后的数据二维ndarray数组，形状：(总窗数, 每窗的特征向量维度)
    :param frame_size: 帧长（即：一帧中包含的数据窗的长度）变量，整数标量。
    :return: 分帧后的3维ndarray数组，形状为：(帧数, 每帧的窗数, 每窗的特征向量维度)
    """
    result = []
    for i in range(len(feature_matrix) - frame_size + 1):
        result.append(feature_matrix[i:i + frame_size])
    frames = np.array(result)
    return frames


def make_label(feature_matrix, frame_size=10):
    """
    提取标签。定义每帧的标签是该帧后的第1号窗。注意最后一帧是没有标签的。
    :param feature_matrix: 分窗后的2维ndarray数组，形状：(总窗数, 每窗的特征向量维度)
    :param frame_size: 帧长，标量
    :return: 标签2维ndarray数组，形状：(帧数-1, 每窗的特征向量维度)
    """
    return feature_matrix[frame_size:]


def make_batch(frames: np.ndarray, labels: np.ndarray, batch_size=20) -> tuple[np.ndarray, np.ndarray]:
    """
    分批函数，将所有帧分为若干批，每批包含batch_size个帧。
    :param frames: 分帧后的3维ndarray数组，形状：(帧数, 每帧的窗数, 每窗的特征向量维度)
    :param labels: 标签2维ndarray数组，形状：(帧数, 每窗的特征向量维度)
    :param batch_size: 批量处理大小（即：每批次处理多少个数据帧），整数标量
    :return: 元组。batches 形状：(批次数, 每批的帧数【batch_size】, 每帧的窗数, 每窗的特征向量维度)；
    labels_batched 形状：(批次数, 每批的帧数【batch_size】, 每窗的特征向量维度)
    """
    batches = []
    labels_batched = []
    max_frame_index = (frames.shape[0] // batch_size) * batch_size - 1  # 可能会舍弃一些帧以适应batch
    for i in range(0, max_frame_index, batch_size):
        batches.append(frames[i:i + batch_size])
        labels_batched.append(labels[i:i + batch_size])
    batches, labels_batched = np.array(batches), np.array(labels_batched)
    return batches, labels_batched


# Transformer预测器
# 模型输入参数形状：x:(每批的帧数, 每帧的窗数, 每窗的特征向量维度)
# 模型的输出形状：x:(每批的帧数, 每窗的特征向量维度)
class transformer_forecaster(nn.Module):
    def __init__(self, feature_vector_len, num_heads, num_blocks, category2filename_dict,
                 transformer_forecaster_drop_prob=transformer_forecaster_drop_prob, static_vector_len=7,
                 total_number_categories=49):
        super(transformer_forecaster, self).__init__()
        self.blocks = nn.ModuleList([TransformerBlock(feature_vector_len, num_heads) for _ in range(num_blocks)])
        self.forecast_head = nn.Sequential(nn.Linear(feature_vector_len - static_vector_len, 2 * feature_vector_len),
                                           nn.LeakyReLU(),
                                           nn.Dropout(transformer_forecaster_drop_prob),
                                           nn.Linear(2 * feature_vector_len, feature_vector_len - static_vector_len)
                                           ).to(device=device)
        self.emb = nn.Embedding(total_number_categories, static_vector_len - 5, device=device)
        self.category2filename_dict = category2filename_dict

    def forward(self, x, category, static_vector_len=7):
        emb_tensor = torch.tensor([category], device=device).int()
        emb_out = self.emb(emb_tensor)[0]
        emb_out = emb_out.unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], static_vector_len - 5)
        x = torch.cat((x, emb_out), dim=2)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = x[:, :-static_vector_len]  # 注意！此处去除稳态特征！
        x = self.forecast_head(x)
        return x


# 注：categories是一个一维数组，里面存储的是整数，长度是批次数量(=batches.shape[0])，用于标定当前批电器的种类。
def train(batches: np.ndarray, labels_batched: np.ndarray, categories: list[int], category2filename_dict: dict, num_heads=5, num_blocks=5, lr=0.0001, epochs=6,
          static_vector_len=7, total_number_categories=28, using_mpt=False) -> nn.Module:
    """
    函数功能：模型训练。
    :param batches: 经过批处理后的4维特征数据向量，形状：(批次数, 每批的帧数【batch_size】, 每帧的窗数, 每窗的特征向量维度)
    :param labels_batched: 经过批处理后的3维标签向量，形状：(批次数, 每批的帧数【batch_size】, 每窗的特征向量维度)
    :param categories: 表示每批中电器类型的数字编码，一维列表list，形状：(批次数,)；元素均为整数。
    :param category2filename_dict: 表示电器分类数字编码与电器文本描述映射关系的字典。字典，键：电器类型数字编码；值：文本描述（如：“热水器”）
    :param num_heads: 注意力头数量。标量，整数
    :param num_blocks: Transformer Block的数量。标量，整数。
    :param lr: 学习率。标量，浮点数。
    :param epochs: 迭代次数。标量，整数。
    :param static_vector_len: 静态特征(稳态特征)向量长度，包括了5个固定特征与n=2个词嵌入向量维度。标量，整数。
    :param total_number_categories: 电器类别总数。标量，整数。
    :param using_mpt: 是否使用混合精度训练(MPT)。布尔值。（注意：当显卡支持MPT时才可启用，否则可能会报错。Volta和Ampere架构的GPU都可以。）
    :return: PyTorch模型对象。
    """
    printlog("使用混合精度训练" if using_mpt else "未开启混合精度训练")
    printlog("开始模型训练")
    model = transformer_forecaster(batches.shape[-1] + (static_vector_len - 5), num_heads, num_blocks,
                                   category2filename_dict=category2filename_dict, static_vector_len=static_vector_len,
                                   total_number_categories=total_number_categories)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用SGD或ADAM算法
    model.train()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.41)
    if using_mpt:
        scaler = GradScaler()
        use_amp = True
    else:
        scaler = None
        use_amp = False
    batches, labels_batched = torch.tensor(batches, device=device), torch.tensor(labels_batched, device=device)
    # Training loop
    for epoch in range(epochs):
        train_loss = 0
        counter = 0
        for batch_num in tqdm(range(batches.shape[0]), mininterval=2, file=sys.stdout):
            tensor_train, label_train = batches[batch_num], labels_batched[batch_num]
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    pred = model(tensor_train, categories[batch_num], static_vector_len=static_vector_len)
                    loss = criterion(pred, label_train)
            else:
                pred = model(tensor_train, categories[batch_num], static_vector_len=static_vector_len)
                loss = criterion(pred, label_train)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            train_loss += loss.item()
            counter += 1

        train_loss = train_loss / counter
        scheduler.step()
        print(f'\nEpoch {epoch + 1} training loss: {train_loss}')
    return model


def split_data(data, ratio=0.95) -> tuple[np.ndarray, np.ndarray]:
    """
    函数功能：按顺序划分训练集和测试集。
    :param data: 待分割的ndarray数组，维度不限，若为多维数组，则按照第0维进行划分。
    :param ratio: 训练集占的比例，取值范围：0 < ratio < 1，浮点数标量。
    :return: 二元组。分别是训练集ndarray数组，测试集ndarray数组。
    """
    n = int(len(data) * ratio)
    return data[:n], data[n:]


def calc_static_vector(data: np.ndarray, zero_limited=0.0001) -> np.ndarray:
    """
    传入二维数组，计算每个窗口的稳态特征向量的固定部分，依次包括最大值、波峰系数、有效值、最小值、算术平均值，共5个维度。
    :param data: 分窗后的数据。2维ndarray数组，形状：(窗口数量, 窗长)
    :param zero_limited: 计算有效值时，若出现除以0的情况，以一个极小值zero_limited来代替0。
    :return: 由每窗的特征向量构成的二维数组。形状：(窗口数量, 5)
    """
    max_val = np.max(data, axis=1)
    min_val = np.min(data, axis=1)
    avg_val = np.mean(data, axis=1)
    effective_val = []
    for i in range(data.shape[0]):
        time_step = data[i]
        e_v = max(sqrt(np.mean(time_step ** 2)), zero_limited)
        effective_val.append(e_v)
    effective_val = np.array(effective_val)
    peak_factor = max_val / effective_val
    peak_factor = np.reshape(peak_factor, (-1, 1))
    max_val = np.reshape(max_val, (-1, 1))
    min_val = np.reshape(min_val, (-1, 1))
    avg_val = np.reshape(avg_val, (-1, 1))
    effective_val = np.reshape(effective_val, (-1, 1))
    result = np.hstack((max_val, peak_factor, effective_val, min_val, avg_val))
    return result


def feature_to_series(data: np.ndarray, is_fft=True, f2s=True) -> np.ndarray:
    """
    传入二维数组，计算每个窗口的暂态特征或相反（根据暂态特征逆推时间序列）。
    :param data: 加窗数据，二维ndarray数组，形状：(窗口数量, 窗长)
    :param is_fft: 是否使用FFT求算特征。若不是，则直接返回数据本身作为特征。
    :param f2s: 函数的工作模式是否为由特征逆推时序。True=特征->时序；False=时序->特征。
    :return: 由特征向量构成的矩阵。形状：(窗口数量, 特征维度)
    """
    if not is_fft:
        return data
    else:
        if f2s:
            n = data.shape[1] // 2
            real_coeffs = data[:, :n]
            imag_coeffs = data[:, n:]
            fft_result = real_coeffs + 1.0j * imag_coeffs
            fft_result_conjugate = np.flip(real_coeffs - 1.0j * imag_coeffs, axis=1)
            fft_result_padded = np.hstack(
                (np.zeros((fft_result.shape[0], 1)), fft_result, fft_result_conjugate))  # 将直流分量补回去
            data_restored = np.fft.ifft(fft_result_padded * fft_result_padded.shape[1], axis=1)  # 进行逆FFT变换
            return data_restored
        else:
            if data.shape[1] % 2 == 0:
                raise Exception("窗口长度必须为奇数。")
            static_vector = calc_static_vector(data)
            # label_vector = np.full((data.shape[0], 1), label_index)
            fft_result = np.fft.fft(data, axis=1)[:, 1:] / data.shape[1]  # 去除直流分量
            fft_result = fft_result[:, :int(data.shape[1] / 2)]
            real_coeffs = np.real(fft_result)
            imag_coeffs = np.imag(fft_result)
            # result = np.hstack((real_coeffs, imag_coeffs, static_vector, label_vector))  # 把静态特征加到后方
            result = np.hstack((real_coeffs, imag_coeffs, static_vector))  # 把静态特征加到后方
            return result


def model_test(model: nn.Module, data_test: np.ndarray, batch_index=0, batch_inner_index=5) -> None:
    """
    模型测试函数：可视化模型预测的结果。
    :param model: PyTorch模型对象。
    :param data_test: 一维ndarray时序数据。
    :param batch_index: 选用第batch_index批进行测试。
    :param batch_inner_index: 该批中选用第batch_inner_index帧进行测试。
    :return: None
    """
    model.eval()
    l = nn.L1Loss()
    with torch.no_grad():
        feature_matrix = window_data(data_test, window_size=21)
        feature_matrix = feature_to_series(feature_matrix, True, False)
        frames = make_frame(feature_matrix, frame_size=10)
        labels = make_label(feature_matrix, frame_size=10)
        frames = frames[:-1]  # 最后一帧无标签
        batches, labels_batched = make_batch(frames, labels, batch_size=10)
        batches, labels_batched = torch.Tensor(batches[batch_index]), torch.Tensor(labels_batched[batch_index])
        pred = model(batches)
        loss = l(pred, labels_batched)
        print("TOTAL LOSS: ", loss.item())

        # 如果使用FFT
        pred = feature_to_series(pred.numpy(), True, True)
        labels_batched = feature_to_series(labels_batched.numpy(), True, True)
        pred = pred[batch_inner_index]
        labels_batched = labels_batched[batch_inner_index]

        pred = pred + (np.mean(labels_batched) - np.mean(pred))
        x = range(len(pred))
        plt.plot(x, pred, label='y_pred')
        plt.plot(x, labels_batched, label='y_test')
        plt.title('Two Lines Plot')
        plt.legend()
        plt.show()


def usage(model: nn.Module, data: np.ndarray, label: str, window_size=21, dc_offset_window=1, static_vector_len=7) -> np.ndarray:
    """
    对预测模型的使用作了封装
    :param model: 模型对象。
    :param data: 一维ndarray时序数组。
    :param label: 电器类型文本标签（如：“热水器”）
    :param window_size: 时间窗长，整型标量。
    :param dc_offset_window: 补充DC时，参照之前时间数据的时间窗数量。
    :param static_vector_len: 暂态特征总维度（固定维度=5 + 嵌入维度n=2）。
    :return: 预测的时间序列。一维ndarray数组，长度为1个时间窗。
    """
    data, data_mean, data_std = scale_normal(data)
    category2filename_dict = model.category2filename_dict
    for i in category2filename_dict.items():
        if label == i[1]:
            index = i[0]
            break
    w_data = window_data(data, window_size)
    f_data = feature_to_series(w_data, True, False)  # 如果使用FFT
    b_data = np.array([f_data])
    input_data = torch.Tensor(b_data)
    pred = model(input_data, index, static_vector_len=static_vector_len).detach().numpy()
    pred_data = feature_to_series(pred, True, True)[0].real
    dc_s = w_data[-dc_offset_window:].flatten()
    dc_mean = np.mean(dc_s)
    pred_data = pred_data + (dc_mean - np.mean(pred_data))
    pred_data = pred_data * data_std + data_mean
    return pred_data


def downsample_array(arr):
    downsampled_arr = []
    for i in range(2, len(arr) - 2, 5):
        downsampled_arr.append(arr[i])
    downsampled_arr = np.array(downsampled_arr)
    return downsampled_arr


def read_csv_files(directory: str) -> dict:
    """
    读取一个目录下的所有CSV文件名。
    :param directory: 目录路径。字符串。
    :return: 字典：键为0~n的整数，值为去掉路径及拓展名(.csv)的文件名。
    """
    printlog("正在读取数据集文件")
    csv_files = {}
    count = 0
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            filename = os.path.splitext(file)[0]
            csv_files[count] = filename
            count += 1
    return csv_files


def data_process(directory: str) -> tuple[np.ndarray, np.ndarray, list, dict]:
    """
    数据处理大一统函数：输入存储CSV文件的目录路径，输出含有训练数据、标签、类别的3元组，以及category2filename_dict字典。前三者可直接输入模型中。
    :param directory: 包含多个CSV格式数据集的目录路径。字符串。
    :return: batches: 经过批处理后的4维特征数据向量，形状：(批次数, 每批的帧数【batch_size】, 每帧的窗数, 每窗的特征向量维度)
    labels_batched: 经过批处理后的3维标签向量，形状：(批次数, 每批的帧数【batch_size】, 每窗的特征向量维度)
    categories: 表示每批中电器类型的数字编码，一维列表list，形状：(批次数,)；元素均为整数。
    category2filename_dict: 表示电器分类数字编码与电器文本描述映射关系的字典。字典，键：电器类型数字编码；值：文本描述（如：“热水器”）
    """
    printlog("开始数据预处理")
    data_tensor_arr, label_tensor_arr, category_arr = [], [], []
    category2filename_dict = read_csv_files(directory=directory)
    for index in tqdm(category2filename_dict):
        filename = category2filename_dict[index]
        data = get_data(os.path.join(directory, filename + ".csv"), skip_header=1, usecol=4)
        data = scale_normal(data)[0]
        feature_matrix = window_data(data, window_size=21)
        feature_matrix = feature_to_series(feature_matrix, True, False)  # 如果使用FFT
        frames = make_frame(feature_matrix, frame_size=10)
        labels = make_label(feature_matrix[:, :-5],
                            frame_size=10)  # 此处注意！注意！标签里除去了后面的静态特征，长度为5不是7(因为Embedding向量不在label里面)。
        frames = frames[:-1]  # 最后一帧无标签，不可用于训练
        batches, labels_batched = make_batch(frames, labels, batch_size=10)
        data_tensor_arr.append(batches)
        label_tensor_arr.append(labels_batched)
        category_arr = category_arr + [index] * batches.shape[0]
    data_total_arr = np.concatenate(tuple(data_tensor_arr), axis=0)
    label_total_arr = np.concatenate(tuple(label_tensor_arr), axis=0)
    return data_total_arr, label_total_arr, category_arr, category2filename_dict


if __name__ == "__main__":
    printlog("正在使用设备：" + str(device).upper())
    
    # 模型训练：
    batches, labels_batched, categories, category2filename_dict = data_process("./dataset")
    model = train(batches, labels_batched, categories, category2filename_dict, num_heads=27, num_blocks=10, lr=0.0000325, epochs=35, static_vector_len=7, total_number_categories=21)
    torch.save(model, "model-large.pt")
    printlog("训练完成，模型权重已成功保存。")

    # 模型使用：
    # model = torch.load("model-large.pt", map_location=torch.device(device=device))
    # data = get_data("./dataset/热水壶+电磁炉.csv", usecol=4)[50:140]
    # data_train, data_test = data[:-20], data[-20:]
    # pred = usage(model, data_train, "热水壶+电磁炉")

    # x_test = range(len(data))
    # x_pred = range(len(data) - len(data_test), len(data) + 1)
    # plt.plot(x_test, data, label='y_test')
    # plt.plot(x_pred, pred, label='y_pred')
    # plt.title('Two Lines Plot')
    # plt.legend()
    # plt.show()

