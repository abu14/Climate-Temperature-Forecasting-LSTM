## 🌡️ **Time Series Forecasting for Climate Data using LSTM**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.12+](https://img.shields.io/badge/TensorFlow-2.12+-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)

This project focuses on climate forecasting using the Jena Climate dataset, which includes 14 atmospheric variables recorded every 10 minutes from January 1, 2009, to December 31, 2016. Leveraging Long Short-Term Memory (LSTM) networks, the model aims to predict temperature based on historical data.

#### **Key features include:**
* Data cleaning and feature engineering to enhance model performance.
* Utilization of LSTM for time-series forecasting.
* Achieved a loss of 0.7845 on the training set and 0.0653 on the validation set, indicating strong predictive capability.
* The analysis provides insights into the relationships between different climatic variables, offering valuable information for climate-related studies.

> Refer to the notebook [Here](https://github.com/abu14/Climate-Temperature-Forecasting-LSTM/blob/main/notebooks/Time_Series_Climate_Forecasting_using_LSTM.ipynb) for more detail.

<p align="center">
  <img src="assets/prediction_performance.png" alt="Digit Recognition">
  
</p>


### 🔧 **Tools Used**

<p>
<img src="https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/-TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">  
<img src="https://img.shields.io/badge/-Keras-D00000?style=flat&logo=keras&logoColor=white"> 
<img src="https://img.shields.io/badge/-scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white">
<img src="https://img.shields.io/badge/-NumPy-013243?style=flat&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/-Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white">
<img src="https://img.shields.io/badge/-Seaborn-3888E3?style=flat&logo=seaborn&logoColor=white">
</p>



### 📦 **Installation**

#### Prerequisites
* numpy
* pandas
* seaborn
* matplotlib
* plotly
* scikit-learn
* tensorflow




## 📂 Project Structure
```
project-root/
├── data/             
├── models/             
├── notebooks/
├── src/                
│   ├── data_processing.py
│   ├── features.py
│   ├── modeling.py
│   └── visualize.py
└── scripts/           
```



## 🧠 Model Architecture

```python
Sequential(
    LSTM(32, return_sequences=True, input_shape=(look_back, n_features)),
    Dropout(0.2),
    ReLU(),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1)
)
```



## 📄 License
Distributed under the MIT License. See LICENSE for more information.

## 🙏 Acknowledgments
Jena Climate Dataset provided by Max Planck Institute


<!-- CONTACT -->
## **Contact**

##### Abenezer Tesfaye

⭐️ Email - tesfayeabenezer64@gmail.com
 
Project Link: [Github Repo](https://github.com/abu14/Climate-Temperature-Forecasting-LSTM)
