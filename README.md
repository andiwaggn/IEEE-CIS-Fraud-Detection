# IEEE-CIS-Fraud-Detection
基于IEEE-CIS欺诈检测数据集的机器学习解决方案

## 项目简介
本项目基于IEEE-CIS欺诈检测数据集，通过机器学习算法构建模型，实现欺诈检测。项目包含数据预处理、特征工程、模型训练、模型评估和模型预测等步骤。

## 数据集
数据集来源于IEEE-CIS欺诈检测挑战赛，包含约60万条交易记录，其中约3.5%的交易为欺诈交易。数据集包含多个特征，包括交易金额、交易时间、交易地点、交易设备等。

## 算法

本项目采用多种机器学习算法进行模型训练，包括：
- LightGBM
- XGBoost
- CatBoost

## 使用方法

首先将数据集放入`input/`目录下，然后运行以下脚本：

数据处理：
运行`data_processing.py`脚本，对原始数据进行预处理和特征工程。处理结果保存在`processed/`目录下。

模型训练：
运行`modeling.py`脚本，训练LightGBM、XGBoost和CatBoost模型。
运行`ensemble_modeling.py`脚本，训练集成模型。
处理结果保存在`results/`目录下。