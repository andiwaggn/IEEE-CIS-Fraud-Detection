import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold
from sklearn import metrics
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

data_path='processed/'
output_path='results/'

# 创建输出文件夹
os.makedirs(output_path, exist_ok=True)

print("读取数据")
train_df = pd.read_csv(data_path+'train_processed.csv')
test_df = pd.read_csv(data_path+'test_processed.csv')
submission = pd.read_csv(data_path+'sample_submission.csv')

# 读取特征
f = open(data_path+'feature_columns.txt', 'r')
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

models = {}
results = {}
rounds = 5000
verbose = 500
# 5折交叉验证
cv = KFold(n_splits=5, shuffle=True, random_state=0)

def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc

# LightGBM训练
def train_lgb():
    params = {
        'num_leaves': 256,
        'min_child_samples': 79,
        'objective': 'binary',
        'max_depth': 13,
        'learning_rate': 0.03,
        "boosting_type": "gbdt",
        "subsample_freq": 3,
        "subsample": 0.9,
        "bagging_seed": 11,
        "metric": 'auc',
        "verbosity": -1,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'colsample_bytree': 0.9,
    }
    
    print('开始训练LightGBM')
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []
    lgb_models = []
    
    for i, (trn_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f'Fold {i+1}')
        X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        
        lgb_model = lgb.LGBMClassifier(**params, n_estimators=rounds, n_jobs=-1)
        lgb_model.fit(X_trn, y_trn, 
                     eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(200), lgb.log_evaluation(verbose)])
        
        # 在测试集的预测结果
        val_pred = lgb_model.predict_proba(X_val)[:, 1]
        # 对未知数据的预测
        test_pred = lgb_model.predict_proba(X_test, num_iteration=lgb_model.best_iteration_)[:, 1]
        
        oof_preds[val_idx] = val_pred
        test_preds += test_pred / 5
        
        score = metrics.roc_auc_score(y_val, val_pred)
        scores.append(score)
        print(f'Fold {i+1} AUC: {score:.4f}')
        
        lgb_models.append(lgb_model)
    
    final_score = metrics.roc_auc_score(y, oof_preds)
    print(f'LGB CV: {np.mean(scores):.4f} ± {np.std(scores):.4f}')
    print(f'LGB OOF: {final_score:.4f}')
    
    models['lgb'] = lgb_models
    results['lgb'] = {
        'oof': oof_preds,
        'test': test_preds,
        'cv_scores': scores,
        'oof_score': final_score
    }

# XGBoost训练
def train_xgb():
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'random_state': 0,
        'verbosity': 0
    }
    
    print('开始训练XGBoost')
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []
    xgb_models = []
    
    for i, (trn_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f'Fold {i+1}')
        X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        
        dtrain = xgb.DMatrix(X_trn, y_trn)
        dval = xgb.DMatrix(X_val, y_val)
        dtest = xgb.DMatrix(X_test)
        
        xgb_model = xgb.train(params, dtrain, rounds, 
                             evals=[(dtrain, 'train'), (dval, 'val')],
                             early_stopping_rounds=200, verbose_eval=verbose)
        
        val_pred = xgb_model.predict(dval)
        test_pred = xgb_model.predict(dtest)
        
        oof_preds[val_idx] = val_pred
        test_preds += test_pred / 5
        
        score = metrics.roc_auc_score(y_val, val_pred)
        scores.append(score)
        print(f'Fold {i+1} AUC: {score:.4f}')
        
        xgb_models.append(xgb_model)
    
    final_score = metrics.roc_auc_score(y, oof_preds)
    print(f'XGB CV: {np.mean(scores):.4f} ± {np.std(scores):.4f}')
    print(f'XGB OOF: {final_score:.4f}')
    
    models['xgb'] = xgb_models
    results['xgb'] = {
        'oof': oof_preds,
        'test': test_preds,
        'cv_scores': scores,
        'oof_score': final_score
    }

# CatBoost训练
def train_cat():
    params = {
        'objective': 'Logloss',
        'eval_metric': 'AUC',
        'learning_rate': 0.33,
        'iterations': rounds,
        'early_stopping_rounds': 200,
        'random_seed': 0,
        'verbose': False
    }
    
    print('开始训练CatBoost')
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []
    cat_models = []
    
    for i, (trn_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f'Fold {i+1}')
        X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        
        cat_model = CatBoostClassifier(**params)
        cat_model.fit(X_trn, y_trn, eval_set=(X_val, y_val), verbose=verbose, use_best_model=True)
        
        val_pred = cat_model.predict_proba(X_val)[:, 1]
        test_pred = cat_model.predict_proba(X_test)[:, 1]
        
        oof_preds[val_idx] = val_pred
        test_preds += test_pred / 5
        
        score = metrics.roc_auc_score(y_val, val_pred)
        scores.append(score)
        print(f'Fold {i+1} AUC: {score:.4f}')
        
        cat_models.append(cat_model)
    
    final_score = metrics.roc_auc_score(y, oof_preds)
    print(f'CAT CV: {np.mean(scores):.4f} ± {np.std(scores):.4f}')
    print(f'CAT OOF: {final_score:.4f}')
    
    models['cat'] = cat_models
    results['cat'] = {
        'oof': oof_preds,
        'test': test_preds,
        'cv_scores': scores,
        'oof_score': final_score
    }

# 集成学习
def make_ensemble():
    print('创建集成模型')
    model_names = list(results.keys())
    
    oof_avg = np.mean([results[name]['oof'] for name in model_names], axis=0)
    test_avg = np.mean([results[name]['test'] for name in model_names], axis=0)
    
    ens_score = metrics.roc_auc_score(y, oof_avg)
    print(f'Ensemble OOF: {ens_score:.4f}')
    
    results['ensemble'] = {
        'oof': oof_avg,
        'test': test_avg,
        'oof_score': ens_score
    }

def save_submissions():
    print('保存文件')
    for name, result in results.items():
        sub = submission.copy()
        sub['isFraud'] = result['test']
        sub.to_csv(f'{output_path}sub_{name}.csv', index=False)
        print(f'已保存: sub_{name}.csv, 欺诈率: {(result["test"] > 0.5).mean():.4f}')

# 显示特征重要性
def show_importance():
    if 'lgb' in models:
        print('LightGBM特征重要性前20:')
        lgb_model = models['lgb'][0]
        importance = lgb_model.feature_importances_
        feature_imp = pd.DataFrame({'feature': feature_cols, 'importance': importance})
        feature_imp = feature_imp.sort_values('importance', ascending=False)
        
        for i, row in feature_imp.head(20).iterrows():
            print(f'{row["feature"]}: {row["importance"]:.1f}')
        
        # 绘制重要性图
        plt.figure(figsize=(10, 8))
        top20 = feature_imp.head(20)
        plt.barh(range(len(top20)), top20['importance'])
        plt.yticks(range(len(top20)), top20['feature'])
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(f'{output_path}importance.png', dpi=150)
        plt.close()
        print('特征重要性图已保存')

if __name__ == '__main__':

    train_lgb()
    train_xgb() 
    train_cat()
    make_ensemble()

    for name, result in results.items():
        if 'cv_scores' in result:
            cv_mean = np.mean(result['cv_scores'])
            cv_std = np.std(result['cv_scores'])
            print(f'{name.upper()}: CV {cv_mean:.4f}±{cv_std:.4f}, OOF {result["oof_score"]:.4f}')
        else:
            print(f'{name.upper()}: OOF {result["oof_score"]:.4f}')

    save_submissions()
    show_importance()
    print('完成!')
