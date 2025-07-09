import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
import shutil

# 读取数据
print("开始读取数据...")
train_trans = pd.read_csv('input/train_transaction.csv')
train_id = pd.read_csv('input/train_identity.csv')
test_trans = pd.read_csv('input/test_transaction.csv')
test_id = pd.read_csv('input/test_identity.csv')

print(f"训练集交易数据: {train_trans.shape}")
print(f"训练集身份数据: {train_id.shape}")
print(f"测试集交易数据: {test_trans.shape}")
print(f"测试集身份数据: {test_id.shape}")

# 合并数据
train_data = train_trans.merge(train_id, on='TransactionID', how='left')
test_data = test_trans.merge(test_id, on='TransactionID', how='left')

print(f"合并后训练集: {train_data.shape}")
print(f"合并后测试集: {test_data.shape}")

y = train_data['isFraud']

# 删除TransactionID但保留TransactionDT
drop_cols = ['TransactionID']
train_data = train_data.drop(drop_cols, axis=1)
test_data = test_data.drop(drop_cols, axis=1)

print("开始特征处理")

train_labels = train_data['isFraud']
train_dt = train_data['TransactionDT']
train_features = train_data.drop(['isFraud', 'TransactionDT'], axis=1)
test_dt = test_data['TransactionDT']
test_features = test_data.drop(['TransactionDT'], axis=1)

# 填充缺失值
train_features = train_features.fillna(-999)
test_features = test_features.fillna(-999)

# 找出类别特征
cat_cols = []
for col in train_features.columns:
    if train_features[col].dtype == 'object':
        cat_cols.append(col)

print(f"类别特征: {len(cat_cols)}个")

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    # 把训练和测试数据合一起编码
    all_vals = pd.concat([train_features[col], test_features[col]]).astype(str)
    le.fit(all_vals)
    
    train_features[col] = le.transform(train_features[col].astype(str))
    test_features[col] = le.transform(test_features[col].astype(str))
    le_dict[col] = le

print("标签编码完成")

train_features['TransactionAmt_log'] = np.log1p(train_features['TransactionAmt'])
test_features['TransactionAmt_log'] = np.log1p(test_features['TransactionAmt'])

c_cols = [col for col in train_features.columns if col.startswith('C')]
if len(c_cols) > 0:
    train_features['C_sum'] = train_features[c_cols].sum(axis=1)
    test_features['C_sum'] = test_features[c_cols].sum(axis=1)

# V列的一些统计
v_cols = [col for col in train_features.columns if col.startswith('V')]
if len(v_cols) > 0:
    train_features['V_mean'] = train_features[v_cols].mean(axis=1)
    test_features['V_mean'] = test_features[v_cols].mean(axis=1)
    
    train_features['V_std'] = train_features[v_cols].std(axis=1)
    test_features['V_std'] = test_features[v_cols].std(axis=1)

# 统计缺失值数量
train_features['missing_count'] = (train_features == -999).sum(axis=1)
test_features['missing_count'] = (test_features == -999).sum(axis=1)

print("特征处理完成")

print(f"最终训练集特征形状: {train_features.shape}")
print(f"最终测试集特征形状: {test_features.shape}")
print(f"标签形状: {train_labels.shape}")

final_train = train_features.copy()
final_train['TransactionDT'] = train_dt
final_train['isFraud'] = train_labels

final_test = test_features.copy()
final_test['TransactionDT'] = test_dt

final_train.to_csv('processed/train_processed.csv', index=False)
final_test.to_csv('processed/test_processed.csv', index=False)

shutil.copy('input/sample_submission.csv', 'processed/sample_submission.csv')

feature_cols = train_features.columns.tolist()
with open('processed/feature_columns.txt', 'w') as f:
    for col in feature_cols:
        f.write(f"{col}\n")

print("数据保存完成")
print(f"特征列数: {len(feature_cols)}")

# 简单看看数据分布
print("\n数据基本信息")
print(f"欺诈率: {train_labels.mean():.4f}")
print(f"特征数量: {len(feature_cols)}")
print("\n数据类型分布:")
print(train_features.dtypes.value_counts())
