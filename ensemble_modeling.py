
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold
from sklearn import metrics
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

data_path = 'processed/'
output_path = 'results/'

# 创建输出文件夹
os.makedirs(output_path, exist_ok=True)

print("读取数据")
train_df = pd.read_csv(data_path + 'train_processed.csv')
test_df = pd.read_csv(data_path + 'test_processed.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')

# 读取特征
f = open(data_path + 'feature_columns.txt', 'r')
feature_cols = []
for line in f.readlines():
    feature_cols.append(line.strip())
f.close()

X = train_df.sort_values('TransactionDT')[feature_cols]
y = train_df.sort_values('TransactionDT')['isFraud']
X_test = test_df[feature_cols]

print('训练数据:', X.shape)
print('测试数据:', X_test.shape)
print('特征数:', len(feature_cols))

# 存储模型和结果
models = {}
results = {}
rounds = 3000
verbose = 500
# 5折交叉验证
cv = KFold(n_splits=5, shuffle=True, random_state=0)
def train_bagging_lgb():
    params = {
        'num_leaves': 256,
        'min_child_samples': 79,
        'objective': 'binary',
        'max_depth': 13,
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'subsample_freq': 3,
        'subsample': 0.9,
        'bagging_seed': 11,
        'metric': 'auc',
        'verbosity': -1,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'colsample_bytree': 0.9,
    }
    
    print('开始训练LightGBM集成模型')
    n_models = 3  # 训练3个模型
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []
    lgb_models = []
    
    for i, (trn_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f'Fold {i+1}')
        X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        
        fold_test_preds = []
        for model_i in range(n_models):
            np.random.seed(model_i * 42 + i)
            sample_idx = np.random.choice(len(X_trn), size=int(len(X_trn) * 0.8), replace=False)
            X_sample = X_trn.iloc[sample_idx]
            y_sample = y_trn.iloc[sample_idx]
            
            lgb_model = lgb.LGBMClassifier(**params, n_estimators=rounds, n_jobs=-1, random_state=model_i)
            lgb_model.fit(X_sample, y_sample, 
                         eval_set=[(X_val, y_val)], 
                         callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
            
            test_pred = lgb_model.predict_proba(X_test, num_iteration=lgb_model.best_iteration_)[:, 1]
            fold_test_preds.append(test_pred)
            
            if model_i == 0:
                val_pred = lgb_model.predict_proba(X_val)[:, 1]
                oof_preds[val_idx] = val_pred
            
            lgb_models.append(lgb_model)
        
        # 平均多个模型的预测结果
        test_preds += np.mean(fold_test_preds, axis=0) / 5
        
        score = metrics.roc_auc_score(y_val, oof_preds[val_idx])
        scores.append(score)
        print(f'Fold {i+1} AUC: {score:.4f}')
    
    final_score = metrics.roc_auc_score(y, oof_preds)
    print(f'LGB Bagging CV: {np.mean(scores):.4f} ± {np.std(scores):.4f}')
    print(f'LGB Bagging OOF: {final_score:.4f}')
    
    models['lgb_bagging'] = lgb_models
    results['lgb_bagging'] = {
        'oof': oof_preds,
        'test': test_preds,
        'cv_scores': scores,
        'oof_score': final_score
    }

def train_bagging_xgb():
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'random_state': 0,
        'verbosity': 0
    }
    
    print('开始训练XGBoost集成模型')
    n_models = 3
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []
    xgb_models = []
    
    for i, (trn_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f'Fold {i+1}')
        X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        
        fold_test_preds = []
        for model_i in range(n_models):
            np.random.seed(model_i * 123 + i)
            sample_idx = np.random.choice(len(X_trn), size=int(len(X_trn) * 0.8), replace=False)
            X_sample = X_trn.iloc[sample_idx]
            y_sample = y_trn.iloc[sample_idx]
            
            dtrain = xgb.DMatrix(X_sample, y_sample)
            dval = xgb.DMatrix(X_val, y_val)
            dtest = xgb.DMatrix(X_test)
            
            xgb_model = xgb.train(params, dtrain, rounds, 
                                 evals=[(dval, 'val')],
                                 early_stopping_rounds=200, verbose_eval=0)
            
            test_pred = xgb_model.predict(dtest)
            fold_test_preds.append(test_pred)
            
            if model_i == 0:
                val_pred = xgb_model.predict(dval)
                oof_preds[val_idx] = val_pred
            
            xgb_models.append(xgb_model)
        
        test_preds += np.mean(fold_test_preds, axis=0) / 5
        
        score = metrics.roc_auc_score(y_val, oof_preds[val_idx])
        scores.append(score)
        print(f'Fold {i+1} AUC: {score:.4f}')
    
    final_score = metrics.roc_auc_score(y, oof_preds)
    print(f'XGB Bagging CV: {np.mean(scores):.4f} ± {np.std(scores):.4f}')
    print(f'XGB Bagging OOF: {final_score:.4f}')
    
    models['xgb_bagging'] = xgb_models
    results['xgb_bagging'] = {
        'oof': oof_preds,
        'test': test_preds,
        'cv_scores': scores,
        'oof_score': final_score
    }

def train_bagging_cat():
    params = {
        'objective': 'Logloss',
        'eval_metric': 'AUC',
        'learning_rate': 0.1,
        'iterations': rounds,
        'early_stopping_rounds': 200,
        'random_seed': 0,
        'verbose': False
    }
    
    print('开始训练CatBoost集成模型')
    n_models = 3
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []
    cat_models = []
    
    for i, (trn_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f'Fold {i+1}')
        X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        
        fold_test_preds = []
        for model_i in range(n_models):
            # 随机采样训练数据
            np.random.seed(model_i * 456 + i)
            sample_idx = np.random.choice(len(X_trn), size=int(len(X_trn) * 0.8), replace=False)
            X_sample = X_trn.iloc[sample_idx]
            y_sample = y_trn.iloc[sample_idx]
            
            cat_model = CatBoostClassifier(**params, random_state=model_i)
            cat_model.fit(X_sample, y_sample, eval_set=(X_val, y_val), 
                         verbose=0, use_best_model=True)
            
            test_pred = cat_model.predict_proba(X_test)[:, 1]
            fold_test_preds.append(test_pred)
            
            if model_i == 0:
                val_pred = cat_model.predict_proba(X_val)[:, 1]
                oof_preds[val_idx] = val_pred
            
            cat_models.append(cat_model)
        
        test_preds += np.mean(fold_test_preds, axis=0) / 5
        
        score = metrics.roc_auc_score(y_val, oof_preds[val_idx])
        scores.append(score)
        print(f'Fold {i+1} AUC: {score:.4f}')
    
    final_score = metrics.roc_auc_score(y, oof_preds)
    print(f'CAT Bagging CV: {np.mean(scores):.4f} ± {np.std(scores):.4f}')
    print(f'CAT Bagging OOF: {final_score:.4f}')
    
    models['cat_bagging'] = cat_models
    results['cat_bagging'] = {
        'oof': oof_preds,
        'test': test_preds,
        'cv_scores': scores,
        'oof_score': final_score
    }

def make_final_ensemble():
    print('创建最终集成模型')
    model_names = list(results.keys())

    oof_avg = np.mean([results[name]['oof'] for name in model_names], axis=0)
    test_avg = np.mean([results[name]['test'] for name in model_names], axis=0)
    
    ens_score = metrics.roc_auc_score(y, oof_avg)
    print(f'Final Ensemble OOF: {ens_score:.4f}')
    
    results['final_ensemble'] = {
        'oof': oof_avg,
        'test': test_avg,
        'oof_score': ens_score
    }

def save_submissions():
    print('保存预测文件')
    for name, result in results.items():
        sub = submission.copy()
        sub['isFraud'] = result['test']
        sub.to_csv(f'{output_path}ensemble_{name}.csv', index=False)
        print(f'已保存: ensemble_{name}.csv, 欺诈率: {(result["test"] > 0.5).mean():.4f}')

# 绘制简单的性能对比图
def plot_performance():
    print('绘制性能对比图')
    model_names = []
    auc_scores = []
    
    for name, result in results.items():
        if 'oof_score' in result:
            model_names.append(name)
            auc_scores.append(result['oof_score'])
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, auc_scores, color=['blue', 'red', 'green', 'orange'])
    plt.title('集成学习模型AUC性能对比')
    plt.ylabel('AUC Score')
    plt.ylim(0.5, 1.0)
    
    # 在柱状图上添加数值标签
    for bar, score in zip(bars, auc_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.4f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_path}ensemble_performance.png', dpi=150)
    plt.close()
    print('性能对比图已保存')

if __name__ == '__main__':
    print("开始训练")

    train_bagging_lgb()
    train_bagging_xgb() 
    train_bagging_cat()
    make_final_ensemble()

    for name, result in results.items():
        if 'cv_scores' in result:
            cv_mean = np.mean(result['cv_scores'])
            cv_std = np.std(result['cv_scores'])
            print(f'{name}: CV {cv_mean:.4f}±{cv_std:.4f}, OOF {result["oof_score"]:.4f}')
        else:
            print(f'{name}: OOF {result["oof_score"]:.4f}')

    save_submissions()
    plot_performance()
    print('集成学习完成!')
