<p align="center">
  <br>
  <img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/a32226e8-4ae0-4377-b2f5-e926580a7ed8" style="width:28%;">
</p>
<br>

<h2>
  Power Load Classification and Prediction System Based on Deep Learning Algorithms <br>
  基于深度学习算法的电力负载分类与预测系统
</h2>

### 项目简介

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;以数字经济助力节能减排，以高新技术助力降本增效，是各行各业千家万户的大势所趋。然而，我国目前的电力资源的不合理利用的现象仍然非常严重，有待我们去解决。中国节能产品认证中心的调查发现，一个普通城市家庭平均每天的家电待机时都会造成一定能耗。全国4亿台彩电一年的待机耗电量就高达29.2亿度，相当于大亚湾核电站全年1/3的发电量。因此，以电力监测的方法发掘、控制电力资源浪费的必要性日益凸显，而非侵入式技术，则又是前者的重中之重。我们小组研发的这款基于非侵入式技术的电力负载分类分解与功率预测系统便是利用人工智能技术为家庭和小型企业减少用电开销、推动节约电力资源的真实写照。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;基于上述背景，我们设计出了一种面向家庭用户和小规模企业的电力负载分类与预测系统。其中，“分类”功能可实时告知用户各电器设备的运作状态；“预测”功能则能预估用户未来短时间内的电能消耗。无论是普通家庭还是企业工厂，都亟需这样的一套智能算法和系统来监控电器的实时运行状态，识别不必要的电力消耗。这不仅有助于节约能源，还能降低用电成本。预测短期电能消耗的功能还能使企业更有效地规划电力使用，特别是在实行阶梯电价机制的情况下。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;家庭用户登录我们的系统后，可以查看住宅中实时耗电功率的波形曲线图、近50分钟房屋内消耗的总电能，以及模型判断正在运行的电器（如：“热水器”，“电脑”，或者是多种电器的组合等等）。用户还可以查看AI模型推测后20分钟房屋内的电能消耗情况（预测的功率曲线）。

### 竞品分析

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;面对日益增大的电力系统规模，现仍存在的侵入式电力负载监测产品无力解决，只能日渐淘汰。目前市面上的相关产品主要使用的非侵入式分类分解算法有基于传统机器学习的决策树、随机森林算法、SVM（支持向量机）等，或是利用集成学习算法对其进行集成。负载预测算法主要有回归分析算法、ARIMA算法和基于深度学习的LSTM算法等。以上算法的预测准确率都不理想，并且不具有电力负载场景的针对性——它们只是解决了普通的时间序列预测或分类问题，而没有考虑电功率数据特有的一些性质和处理方法。以下是对于完全相同训练集和测试集下的，基于ARIMA算法、LSTM算法以及我们的系统的预测功率曲线图：

<img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/8d73555e-64fe-4597-bfb9-5e2b85530933" style="width: 80%;" />


### 系统设计思路

- **AI模型：** 采用级联模型结构，将任务分为分类分解与预测两个级联的模块。分类采用深度全连接神经网络模型去处理，根据时间窗的特征，实现分类；分解任务通过分类任务去实现；预测模型受生成式语言模型的启发而设计，采用了以`Transformer`结构和为核心的深度神经网络两个模型通过电器类型标签相关联。

- **系统架构：** 采用适于现代化超算服务平台的“应用分布，算力集中”云端协同架构。包含边缘设备（电能表）、分布式的后端和数据库、位于超算中心的分类与预测AI模型，以及位于移动设备的微信小程序客户端。这样的设计可以在一定程度上解决前文所提及的大规模用户并发请求造成模型无法及时推理的问题。

- **前端与后端设计：** 我们主要使用了微信开发者工具进行前端开发，波形绘制使用了ECharts库中提供的折线图组件。前台轮询查询数据。为提高性能，后端程序使用C++17构建Server，其基于单Reactor多线程架构。服务器系统核心由日志记录、线程池管理、IO多路复用、HTTP处理、缓冲区管理和阻塞队列等模块组成。HTTP请求报文通过分散读方式读入，并利用有限状态机与正则表达式进行高效解析。

<table>
  <thead>
    <tr>
      <th>模型工作流图</th>
      <th>系统宏观架构</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/94326052-6266-42b5-b519-80ea3f5eb622" style="width: 65%;"/></td>
      <td><img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/246ece1e-53d2-4eaf-9263-dbed7752608b" style="width: 65%;"/></td>
    </tr>
  </tbody>
</table>

### 模型详细设计

- **分类分解模型：** 特征工程方面：先将输入的数据加窗处理，再将功率数值进行分组。接着，统计每个时间窗内落入各组的样本数量，得到频数分布。为了得到特征向量，我们对每个时间窗的频数进行归一化处理，即除以窗长。最后，使用PCA算法进行特征矩阵压缩。模型设计方面：选用全连接神经网络`FNN`，输入该时间窗对应的特征向量，输出对应电器类别的独热编码。我们使用3层全连接神经网络模型(28, 14, 7) 进行数据分类任务（7种单电器）；使用3层全连接神经网络模型(84, 42, 21) 进行功率标签分解任务（21种电器两两组合）。选用模型时，我们对比了全连接网络与`GRU`（循环门控单元）的性能。经过超参数的调整，两者分别可达到91%和90%的最高准确率，也就是说，两者的效果都比较好并且相差无几。但是经过实验与上线测试说明，`GRU`由于要关注其他时间窗，其推理功耗明显大于全连接网络。另一方面，假定每个时间窗之间是互相时序不相关的，这样模型可以专注的处理每一个时间窗中的特征而做出判断，并将标签数据传给预测模型的嵌入层进一步处理，预测模型再利用时序特性进行预测，各司其职。综上两种原因而选择采用全连接网络。以下是分类模型结构示意图：


<img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/b4d5f10e-2e82-4483-9029-08061f71b341" style="width: 70%;" />

<br><br>

- **负载功率预测模型：** 对于电力功率时间序列预测的任务，我们设计了如下模型结构。在处理数据时，首先对数据进行分帧处理，然后在每个数据帧内进行分窗。每个时间窗都会计算出一个特征向量，多个时间窗的特征向量组成一个特征矩阵。这些特征向量由波动大的暂态特征和不易变的稳态特征两部分拼接而成。暂态特征的组成是通过快速傅里叶变换得到的各频率序数对应的振幅和相位（由于直流分量(DC)不包含任何有效信息，而DC所能反映的功率波形信号的采样均值可通过稳态特征体现，故我们将其去除）；而根据电学相关理论，稳态特征则由最大值、最小值、算术平均值、能量有效值和波峰系数等构成。此外，模型通过Embedding层将离散的电器种类映射为连续的嵌入向量，作为稳态特征的重要组成部分。受生成式语言模型的启发，特征矩阵被视为一句话的“词嵌入矩阵”，通过5~6层`Transformer Block`（最终选定了6层）和后续的注意力平均化处理，生成预测向量（相当于LM中生成的token对应的词嵌入向量）。最后，通过逆FFT将预测向量中的暂态特征（频域向量）转换为时间序列，从而完成功率时序预测。以下是功率数值预测模型示意图：

<img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/acbe5aed-2351-4396-9afe-27cabb55fbf0" style="width: 70%;" />

### 模型在实验数据集中的效果展示

所用数据集采集自真实用电设备，数据量200万分钟，时间粒度1分钟，图中预测任务为给定100分钟数据，预测未来20分钟。下图所示数据均已经过标准化处理。该数据集涉及企业秘密，不作开源。

<img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/91d36ba6-9291-4e11-a3ad-4159cb515afa" style="width: 55%;">

### 部署与使用方法

- **客户端：** 项目使用微信小程序作为客户端前端，只需将`WeChat_client`目录用微信开发者工具打开，然后发布或使用第三方工具打包即可。
- **分布式后端（客户端服务）：** 项目使用`C++ 17`标准开发客户端服务模块后端，该程序针对`Linux`系统开发，不跨平台。使用CMake在Linux系统下构建二进制可执行文件。进入CPP_backend目录，然后执行以下参考命令：
  
```bash
mkdir build
cd build
cmake ..
make
```

在build目录下即会生成可执行文件`webserver`，运行时需确保`SQLite`数据库文件`powerload.db`文件位于与`webserver`相同的目录下。若不方便使用C++后台，可以使用基于Python Flask的后台代替，两者功能上完全等效，但是效率略有降低。(文件位置：`/Python_backend/app.py`)

- **分布式后端（数据计算）与 AI模型计算中心端** 该部分代码位于`Python_backend`目录下。提供基于Flask HTTP通信方案的代码组合(`app-backend.py`与`app-ai.py`)与基于`Rabbit MQ`的通信方案代码组合(`app-backend-mq.py`与`app-ai-mq.py`)，可以选择其中一组成组运行，建议先启动AI模型端，再启动分布式后端。分类分解与预测模型二进制文件分别位于classify目录与prediction目录下，后端会自动调用这些模型。
- **模型训练：** 确保`dataset`目录存在于`Python_backend`目录下，`dataset`目录中包含若干`CSV`文件，该文件的文件名应为标签名(对应电器或电器组合的名称)。
  
执行以下命令可以分别训练分类分解模型(用时：CPU环境3分钟左右)与数值预测模型(用时：CPU环境2小时左右)。
```bash
Python ./classify/Classification_Train_v6.3.py
Python ./classify/prediction.py
```

可以根据数据集特点与其他情况，自行调整超参数，修改代码：

  - 数值预测模型训练代码(`prediction.py`)中，以下位置的代码可能需要改动：
```python
# 下列代码位于主函数
batches, labels_batched, categories, category2filename_dict = data_process("./dataset")
model = train(batches, labels_batched, categories, category2filename_dict, num_heads=27, num_blocks=6, \
        lr=0.00004, epochs=13, static_vector_len=7, total_number_categories=21)
torch.save(model, "model-large.pt")

# 下列代码位于data_process函数
data = get_data(os.path.join(directory, filename + ".csv"), skip_header=1, usecol=4)
```

  - 分类分解模型训练代码(`Classification_Train_v6.3.py`)中，以下位置的代码可能需要改动：
```python
# 下列代码位于文件开头处
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
```

### 客户端GUI示例图

以下示例图中分别展示了用户登录、个人主页、实时数据波形及分类分解，以及功率数值预测波形 共4个界面。

<img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/9eff7d69-38a4-4127-91c4-1282b4c49c84" style="width: 70%;" />

### 项目展示视频

<video src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/0684486d-e391-4301-834f-0c5af6e6ce90" style="width: 65%;"></video>

### 著作权声明

Copyright © 2024 The research and development group for Power Load Classification and Prediction System Based on Deep Learning Algorithm, Faculty of Computer Science & Technology, Qilu University of Technology (Shandong Academy of Sciences).

齐鲁工业大学（山东省科学院）计算机科学与技术学部 “基于深度学习算法的电力负载分类与预测系统项目”研究与开发小组 保留所有权利。

- 研究与开发小组成员名单：
  - 杜宇 Yu DU (齐鲁工业大学（山东省科学院）计算机科学与技术学部, No.202103180009)
  - 姜川 Chuan JIANG (齐鲁工业大学（山东省科学院）计算机科学与技术学部, No.202103180020)
  - 李晓语 Xiaoyu LI (齐鲁工业大学（山东省科学院）计算机科学与技术学部, No.202103180001)
  - 李庆隆 Qinglong LI (齐鲁工业大学（山东省科学院）计算机科学与技术学部, No.202103180027)
  - 张一雯 Yiwen ZHANG (齐鲁工业大学（山东省科学院）计算机科学与技术学部, No.202103180051)
- 指导教师：
  - 贾瑞祥老师 Ruixiang JIA (齐鲁工业大学（山东省科学院）计算机科学与技术学部, 软件工程系讲师)
  - 陈静老师 Jing CHEN (山东省计算中心（国家超级计算济南中心）)

本项目基于我们自定义的开源协议(许可证)开放源代码，在您通过任何方式获得源代码前，请仔细阅读并充分理解许可证的全部内容。许可证文件为LICENSE文件[https://github.com/duyu09/Powerload-Classification-and-Prediction-System/blob/main/LICENSE]

- 其它说明：
  - **本项目已参加2024年第17届中国大学生计算机设计大赛4C2024人工智能实践赛赛道。**
  - 本项目的LOGO由智谱清言`CogView`AI绘图工具绘制，用作LOGO时有修改。LOGO寓意：主体金属环象征仪表盘，我们的项目是非侵入式电力监测系统；环内的条纹象征着电力数据波形，同时它也可看作是城市里林立的高楼大厦，象征着我们的系统可以服务于城市电力系统的运转；背景是木制桌面，象征着我们的系统助力绿色发展、减少碳排放。

### 特别感谢

- <b>齐鲁工业大学 (山东省科学院).</b> (<b>Qilu University of Technology (Shandong Academy of Sciences).</b>)

<img src="https://user-images.githubusercontent.com/92843163/229986960-05c91bf5-e08e-4bd3-89a4-deee3d2dbc6d.svg" style="width:40%;border-radius:20px;">

- <b>计算机科学与技术学部 山东省计算中心(国家超级计算济南中心).</b> (<b>Faculty of Computer Science and Technology. National Supercomputing Center in Jinan.</b>)

<img src="https://github.com/duyu09/Intelligent-Learning-Platform/assets/92843163/3a31a937-804f-4230-9585-b437430ac950" style="width:40%;border-radius:20px;">
<br><br>

<b>齐鲁工业大学 (山东省科学院) 开发者协会.</b> (<b>Developer Association of Qilu University of Technology (Shandong Academy of Sciences).</b>)

<img src="https://github.com/duyu09/Intelligent-Learning-Platform/assets/92843163/7a554ca6-49b8-4099-b214-4c4ceff7c9a3" style="width:40%;border-radius:20px;">

### 友情链接

- 齐鲁工业大学(山东省科学院) https://www.qlu.edu.cn/
  
- 山东省计算中心(国家超级计算济南中心) https://www.nsccjn.cn/

- 齐鲁工业大学(山东省科学院) 计算机科学与技术学部 http://jsxb.scsc.cn/

- DuYu的个人网站 https://www.duyu09.site/

- DuYu的GitHub账号 https://github.com/duyu09/

### 访客统计

<div><b>Number of Total Visits (All of Duyu09's GitHub Projects): </b><br><img src="https://profile-counter.glitch.me/duyu09/count.svg" /></div> 

<div><b>Number of Total Visits (基于深度学习算法的电力负载分类与预测系统 <i><b>PLDA</b></i>): 
</b><br><img src="https://profile-counter.glitch.me/duyu09-PLDA-SYSTEM/count.svg" /></div> 
