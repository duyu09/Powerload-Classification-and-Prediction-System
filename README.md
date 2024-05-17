<p align="center">
  <br>
  <img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/a32226e8-4ae0-4377-b2f5-e926580a7ed8" style="width:28%;">
</p>
<br>


# Power Load Classification and Prediction System Based on Deep Learning Algorithms

#### Language of This Document

[**English**](https://github.com/duyu09/Powerload-Classification-and-Prediction-System/blob/main/README.md) | [**简体中文**](https://github.com/duyu09/Powerload-Classification-and-Prediction-System/blob/main/README_SC.md) | [**Tiếng Việt**](https://github.com/duyu09/Powerload-Classification-and-Prediction-System/blob/main/README_VN.md)

_This document was translated by OpenAI's ChatGPT-4o model and is for reference only. Please refer to the Simplified Chinese version as the authoritative source._

### Project Overview

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Leveraging the digital economy to support energy conservation and emission reduction, and using high-tech to assist in cost reduction and efficiency enhancement, is an inevitable trend for all industries and households. However, the irrational use of power resources in China is still a significant issue that needs to be addressed. According to a survey by the China Energy Conservation Product Certification Center, the standby mode of household appliances in an average urban household consumes a notable amount of energy daily. For instance, the annual standby power consumption of 400 million TVs nationwide is as high as 2.92 billion kWh, equivalent to one-third of the annual electricity generation of the Daya Bay Nuclear Power Station. Thus, the necessity of detecting and controlling power resource wastage through power monitoring is becoming increasingly prominent, with non-intrusive techniques being crucial. Our team developed this non-intrusive power load classification and power prediction system utilizing AI technology to help households and small businesses reduce electricity expenses and promote power resource conservation.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Based on this background, we designed a power load classification and prediction system aimed at household users and small businesses. The "classification" function can inform users about the operational status of various electrical devices in real time, while the "prediction" function can estimate the user's power consumption in the near future. Both ordinary households and factory enterprises urgently need such an intelligent system to monitor the real-time operation of appliances and identify unnecessary power consumption. This not only helps conserve energy but also reduces electricity costs. The short-term power consumption prediction function allows businesses to plan their electricity use more effectively, especially under a tiered electricity pricing system.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Upon logging into our system, household users can view the waveform chart of real-time power consumption in their residence, the total energy consumption of the house in the past 50 minutes, and the model's determination of the operating appliances (such as "water heater," "computer," or a combination of multiple appliances, etc.). Users can also view the predicted power consumption of the house for the next 20 minutes (predicted power curve).

### Competitive Analysis

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; With the increasing scale of power systems, the currently existing intrusive power load monitoring products are unable to address the issue and are gradually being phased out. The related products on the market mainly use non-intrusive classification algorithms based on traditional machine learning such as decision trees, random forests, and SVM (Support Vector Machines), or employ ensemble learning algorithms. The main load prediction algorithms include regression analysis, `ARIMA`, and `LSTM` based on deep learning. The prediction accuracy of these algorithms is not ideal, and they do not cater to the specifics of power load scenarios—they only solve generic time series prediction or classification problems without considering the unique characteristics and processing methods of power data. Below is a comparison of the power prediction curves using ARIMA, LSTM, and our system on the same training and testing sets:

<img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/8d73555e-64fe-4597-bfb9-5e2b85530933" style="width: 80%;" />

### System Design Concept

- **AI Model:** The system adopts a cascading model structure, dividing tasks into two cascading modules: classification and prediction. The classification module uses a deep fully connected neural network model to process the features of the time window and perform classification. The decomposition task is realized through the classification task. The prediction model, inspired by generative language models, utilizes deep neural network models with a Transformer structure, linking them through appliance type labels.

- **System Architecture:** The system employs a "distributed application, centralized computing" cloud collaborative architecture suitable for modern supercomputing service platforms. It includes edge devices (power meters), a distributed backend and database, classification and prediction AI models located at the supercomputing center, and a WeChat mini-program client on mobile devices. This design addresses the issue of model inference delays caused by large-scale concurrent user requests to some extent.

- **Frontend and Backend Design:** The frontend is developed using the `WeChat Developer Tool`, with waveform plotting using the line chart component provided by the `ECharts` library. The frontend polls for data. For performance enhancement, the backend server is constructed using `C++17`, based on a single reactor multithreaded architecture. The server core includes modules for log recording, thread pool management, I/O multiplexing, HTTP handling, buffer management, and blocking queue management. HTTP request messages are read using scatter-gather I/O and efficiently parsed with finite state machines and regular expressions.

| Model Workflow Diagram | System Macro Architecture |
| ------ | ------ |
| <img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/94326052-6266-42b5-b519-80ea3f5eb622" style="width: 65%;"/> | <img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/246ece1e-53d2-4eaf-9263-dbed7752608b" style="width: 65%;"/> |

### Detailed Model Design

- **Classification and Decomposition Model:** In terms of feature engineering, the input data is first windowed, and the power values are grouped. Then, the sample count falling into each group within each time window is calculated to obtain the frequency distribution. To derive the feature vector, the frequency of each time window is normalized by dividing by the window length. Finally, PCA is used to compress the feature matrix. For model design, a fully connected neural network (`FNN`) is selected, inputting the feature vector corresponding to the time window and outputting the one-hot encoding of the corresponding appliance category. We use a 3-layer fully connected neural network model `(28, 14, 7)` for data classification tasks (7 single appliances) and a 3-layer fully connected neural network model `(84, 42, 21)` for power label decomposition tasks (21 combinations of two appliances). When selecting models, we compared the performance of fully connected networks and `GRU` (Gated Recurrent Unit). After hyperparameter tuning, both achieved maximum accuracies of 91% and 90%, respectively, meaning both performed well with minimal difference. However, experiments and online testing indicated that GRU's inference power consumption was significantly higher than the fully connected network. Additionally, assuming no time-sequential correlation between time windows, the model can focus on processing each time window's features to make judgments and pass the label data to the prediction model's embedding layer for further processing. The prediction model then uses time sequence characteristics for prediction, with each model handling its respective tasks. These reasons led to the choice of a fully connected network. Below is the classification model structure diagram:

<img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/b4d5f10e-2e82-4483-9029-08061f71b341" style="width: 70%;" />

<br>

- **Load Power Prediction Model:** For the task of power time series prediction, we designed the following model structure. When processing data, the data is first framed and then windowed within each frame. Each time window calculates a eigenvector, and multiple time windows' eigenvector form a eigenmatrix. These feature vectors consist of transient features with large fluctuations and stable features that are less likely to change. Transient features are composed of the amplitude and phase corresponding to each frequency order obtained through fast Fourier transform (excluding the DC component, as it contains no valid information, which is reflected by the steady-state features via the sampling mean of the power waveform signal). According to electrical theory, steady-state features consist of maximum value, minimum value, arithmetic mean, RMS, and crest factor. The model maps discrete appliance types into continuous embedding vectors through an embedding layer, forming an essential part of the steady-state features. Inspired by generative language models, the eigenmatrix is regarded as a "embedding matrix" of a sentence, processed through 5-6 layers of `Transformer Blocks` (6 layers were ultimately selected) and subsequent attention pooling, generating the prediction vector (analogous to the token embedding vector in LM). Finally, the transient features (frequency domain vectors) in the prediction vector are converted into time series through inverse FFT, completing power time series prediction. Below is the power value prediction model diagram:

<img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/acbe5aed-2351-4396-9afe-27cabb55fbf0" style="width: 70%;" />

### Model Performance on Experimental Dataset

The dataset used is collected from real electrical equipment, with a data volume of 2 million minutes and a time granularity of 1 minute. The prediction task is to predict the power consumption for the next 20 minutes given 100 minutes of data. The data shown below is standardized. This dataset involves business secrets and is not open-source.

<img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/91d36ba6-9291-4e11-a3ad-4159cb515afa" style="width: 55%;">

### Deployment and Usage Instructions

- **Client:** The project uses a WeChat mini program as the client frontend. Simply open the `WeChat_client` directory with the WeChat Developer Tools, then publish or package it using third-party tools.

- **Distributed Backend (Client Service):** The client service module backend is developed using `C++ 17` and is targeted for `Linux` systems only. It is not cross-platform. Use CMake to build the binary executable on a Linux system. Navigate to the CPP_backend directory and execute the following commands:

```bash
mkdir build
cd build
cmake ..
make
```

The executable `webserver` will be generated in the build directory. Ensure the `SQLite` database file `powerload.db` is in the same directory as `webserver` when running. If the C++ backend is not convenient, you can use a Python Flask backend instead, which provides the same functionality but with slightly reduced efficiency. (File location: `/Python_backend/app.py`)

- **Distributed Backend (Data Calculation) and AI Model Calculation Center:** The code for this section is located in the `Python_backend` directory. It provides two communication scheme combinations: a Flask HTTP communication combination (`app-backend.py` and `app-ai.py`) and a `RabbitMQ` communication combination (`app-backend-mq.py` and `app-ai-mq.py`). You can choose either combination to run. It is recommended to start the AI model side first, followed by the distributed backend. The binary files for classification decomposition and prediction models are located in the classify directory and the prediction directory, respectively, and the backend will automatically invoke these models.

- **Model Training:** Ensure the `dataset` directory exists in the `Python_backend` directory, containing several `CSV` files. The filenames should correspond to the labels (names of appliances or appliance combinations).

Execute the following commands to train the classification decomposition model (approximately `3 minutes` on CPU) and the numerical prediction model (approximately `2 hours` on CPU), respectively:

```bash
Python ./classify/Classification_Train_v6.3.py
Python ./classify/prediction.py
```

You can adjust hyperparameters and modify the code according to the characteristics of the dataset and other conditions:

For the numerical prediction model training code (`prediction.py`), the following code sections may need changes:

```python
# The following code is in the main function
batches, labels_batched, categories, category2filename_dict = data_process("./dataset")
model = train(batches, labels_batched, categories, category2filename_dict, num_heads=27, num_blocks=6, \
        lr=0.00004, epochs=13, static_vector_len=7, total_number_categories=21)
torch.save(model, "model-large.pt")

# The following code is in the data_process function
data = get_data(os.path.join(directory, filename + ".csv"), skip_header=1, usecol=4)
```

For the classification decomposition model training code (`Classification_Train_v6.3.py`), the following code sections may need changes:

```python
# The following code is at the beginning of the file
dc = r'../dataset'  # Read all CSV files in this directory
model_save_path = r'Classification-Model-v6.3-2.2.keras'  # Path to save the model file
matrix_save_path = r'Classification_feature-2.2.mat'  # Path to save the feature matrix file
dict_save_path = r"label2index_dict.pkl"  # Path to save the dictionary mapping labels to numerical codes
pca_save_path = r"pca.pkl"  # Path to save the PCA object
testRate = 0.15  # Proportion of the dataset used as the test set 0.15
frameLength = 800  # Frame length 800
step = 800  # Feature extraction step length 800
max_value = 3000000  # Maximum power value 3000000
eps = 75  # Number of neural network training iterations 75
lamb = 0.001  # L1 regularization penalty coefficient 0.001
pca_n_components = 67  # Proportion of variance retained by PCA (information retention level) 0.96
es_patience = 3  # Early stopping threshold (monitoring loss function) 3
power_column = -1  # Field number of power value in the dataset (-1 for the last field)
```

### Client GUI Sample Images

The following sample images display the user login, personal homepage, real-time data waveform and classification decomposition, and power value prediction waveform across four interfaces.

<img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/9eff7d69-38a4-4127-91c4-1282b4c49c84" style="width: 70%;" />

### Project Demonstration Video

<video src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/0684486d-e391-4301-834f-0c5af6e6ce90" style="width: 65%;"></video>

-----

### Copyright Notice

Copyright © 2024 _The research and development group for Power Load Classification and Prediction System Based on Deep Learning Algorithm, Faculty of Computer Science & Technology, Qilu University of Technology (Shandong Academy of Sciences)_.

All rights reserved by the research and development group for the Power Load Classification and Prediction System Based on Deep Learning Algorithm project at the Faculty of Computer Science and Technology, Qilu University of Technology (Shandong Academy of Sciences).

- **Research and development team members**:
  - **Yu DU** (Faculty of Computer Science and Technology, Qilu University of Technology (Shandong Academy of Sciences), No.202103180009)
  - **Chuan JIANG** (Faculty of Computer Science and Technology, Qilu University of Technology (Shandong Academy of Sciences), No.202103180020)
  - **Xiaoyu LI** (Faculty of Computer Science and Technology, Qilu University of Technology (Shandong Academy of Sciences), No.202103180001)
  - **Qinglong LI** (Faculty of Computer Science and Technology, Qilu University of Technology (Shandong Academy of Sciences), No.202103180027)
  - **Yiwen ZHANG** (Faculty of Computer Science and Technology, Qilu University of Technology (Shandong Academy of Sciences), No.202103180051)
- **Supervisors**:
  - **Ruixiang JIA** (Lecturer of Faculty of Computer Science and Technology, Qilu University of Technology (Shandong Academy of Sciences))
  - **Jing CHEN** (Shandong Computer Science Center, National Supercomputing Center in Jinan)

This project is open source under our custom license. Please read and fully understand the content of the license before obtaining the source code by any means. The license file can be found at [LICENSE](https://github.com/duyu09/Powerload-Classification-and-Prediction-System/blob/main/LICENSE).

- Other Notes:
  - **This project has participated in the 17th China Collegiate Computing Competition (4C2024) Artificial Intelligence Practice Track in 2024.**
  - The project's logo was created using the AI drawing tool `CogView` and modified for use as a logo. The logo's meaning: The main metal ring symbolizes a dashboard, indicating our project is a non-intrusive power monitoring system; the stripes within the ring represent power data waveforms and can also be seen as tall buildings in the city, symbolizing that our system can serve the operation of urban power systems; the wooden background represents our system's contribution to green development and carbon emission reduction.
 
-----

### Special Acknowledgement

- **Qilu University of Technology (Shandong Academy of Sciences).**

<img src="https://user-images.githubusercontent.com/92843163/229986960-05c91bf5-e08e-4bd3-89a4-deee3d2dbc6d.svg" style="width:40%;border-radius:20px;">

- **Faculty of Computer Science and Technology, National Supercomputing Center in Jinan, SCSC.**

<img src="https://github.com/duyu09/Intelligent-Learning-Platform/assets/92843163/3a31a937-804f-4230-9585-b437430ac950" style="width:40%;border-radius:20px;">
<br><br>

- **Developer Association of Qilu University of Technology (Shandong Academy of Sciences).**

<img src="https://github.com/duyu09/Intelligent-Learning-Platform/assets/92843163/7a554ca6-49b8-4099-b214-4c4ceff7c9a3" style="width:40%;border-radius:20px;">

### Links

- Qilu University of Technology (Shandong Academy of Sciences): [https://www.qlu.edu.cn/](https://www.qlu.edu.cn/)
  
- Shandong Computer Center (National Supercomputing Center in Jinan): [https://www.nsccjn.cn/](https://www.nsccjn.cn/)

- Faculty of Computer Science and Technology, Qilu University of Technology (Shandong Academy of Sciences): [http://jsxb.scsc.cn/](http://jsxb.scsc.cn/)

- DuYu's Personal Website: [https://www.duyu09.site/](https://www.duyu09.site/)

- DuYu's GitHub Account: [https://github.com/duyu09/](https://github.com/duyu09/)

### Visitor Statistics

<div><b>Number of Total Visits (All of Duyu09's GitHub Projects): </b><br><img src="https://profile-counter.glitch.me/duyu09/count.svg" /></div> 

<div><b>Number of Total Visits (Power Load Classification and Prediction System Based on Deep Learning Algorithms): </b>
<br><img src="https://profile-counter.glitch.me/duyu09-PLDA-SYSTEM/count.svg" /></div> 
