# 📥 数据集下载指南

由于IEEE-CIS欺诈检测数据集文件较大（约1.5GB），我们没有将原始数据包含在GitHub仓库中。请按照以下步骤下载并设置数据集。

## 🎯 数据集信息

- **数据来源**: Kaggle IEEE-CIS Fraud Detection Competition
- **数据集大小**: ~1.5GB (压缩后)
- **文件数量**: 5个CSV文件
- **记录数量**: 训练集 590,540条，测试集 506,691条
- **特征数量**: 434个特征

## 📂 所需文件列表

下载完成后，您需要确保`input/`文件夹包含以下文件：

```
input/
├── train_transaction.csv      (~447MB) - 训练集交易数据
├── train_identity.csv         (~37MB)  - 训练集身份数据  
├── test_transaction.csv       (~374MB) - 测试集交易数据
├── test_identity.csv          (~29MB)  - 测试集身份数据
└── sample_submission.csv      (~12MB)  - 提交样本格式
```

## 🚀 方法一：从Kaggle官网下载（推荐）

### 步骤1：注册Kaggle账号
1. 访问 [Kaggle官网](https://www.kaggle.com/)
2. 点击 "Register" 注册新账号
3. 验证邮箱完成注册

### 步骤2：访问竞赛页面
1. 访问竞赛页面：[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
2. 点击 "Join Competition" 加入竞赛
3. 接受竞赛规则和条款

### 步骤3：下载数据
1. 点击 "Data" 标签页
2. 点击 "Download All" 下载所有文件
3. 或者单独下载需要的文件：
   - `train_transaction.csv`
   - `train_identity.csv` 
   - `test_transaction.csv`
   - `test_identity.csv`
   - `sample_submission.csv`

### 步骤4：解压和放置
1. 解压下载的zip文件
2. 将所有CSV文件放入项目的 `input/` 文件夹
3. 确保文件路径正确

## 🔧 方法二：使用Kaggle API（高级用户）

### 步骤1：安装Kaggle API
```bash
pip install kaggle
```

### 步骤2：配置API密钥
1. 登录Kaggle账号
2. 进入 "Account" → "API" → "Create New API Token"
3. 下载 `kaggle.json` 文件
4. 将文件放置到：
   - **Windows**: `C:\Users\<username>\.kaggle\kaggle.json`
   - **macOS/Linux**: `~/.kaggle/kaggle.json`

### 步骤3：下载数据集
```bash
# 创建input文件夹
mkdir input
cd input

# 下载完整数据集
kaggle competitions download -c ieee-fraud-detection

# 解压文件
unzip ieee-fraud-detection.zip

# 删除zip文件（可选）
rm ieee-fraud-detection.zip
```

## 📋 方法三：使用Python脚本自动下载

创建 `download_data.py` 文件：

```python
import os
import kaggle

def download_ieee_fraud_data():
    """自动下载IEEE欺诈检测数据集"""
    
    # 创建input文件夹
    os.makedirs('input', exist_ok=True)
    
    print("开始下载IEEE-CIS Fraud Detection数据集...")
    
    try:
        # 下载数据集
        kaggle.api.competition_download_files(
            'ieee-fraud-detection', 
            path='input', 
            unzip=True
        )
        print("✅ 数据集下载完成！")
        
        # 检查文件
        required_files = [
            'train_transaction.csv',
            'train_identity.csv', 
            'test_transaction.csv',
            'test_identity.csv',
            'sample_submission.csv'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(f'input/{file}'):
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ 缺少文件: {missing_files}")
        else:
            print("✅ 所有必要文件已就位！")
            
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        print("请检查Kaggle API配置或手动下载数据")

if __name__ == "__main__":
    download_ieee_fraud_data()
```

运行脚本：
```bash
python download_data.py
```
