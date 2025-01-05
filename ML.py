import torch
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, roc_curve
import json
# 加载数据

data_dir = "/mnt/hpc/home/algorithm/xiongjun/NEW-MIL-dir-backup/huiyidata/data2"
exp_dir = "/mnt/hpc/home/algorithm/xiongjun/NEW-MIL-dir-backup/huiyidata/data2/exp"
train_data = torch.load(os.path.join(data_dir, 'train_data_72.pt'))
train_label = torch.load(os.path.join(data_dir, 'train_label_72.pt'))
test_data = torch.load(os.path.join(data_dir, 'test_data_72.pt'))
test_label = torch.load(os.path.join(data_dir, 'test_label_72.pt'))

# 数据预处理
def preprocess_data(data):
    part1 = data[:, :13000]
    part2 = data[:, 100000:113000]
    return torch.cat([part1, part2], dim=1)

X_train = preprocess_data(train_data).numpy()
y_train = np.argmax(train_label.numpy(), axis=1)
X_test = preprocess_data(test_data).numpy()
y_test = np.argmax(test_label.numpy(), axis=1)

# # Calculate sample size for each dataset
# train_sample_size = len(X_train) // 10
# test_sample_size = len(X_test) // 10

# # Stratified sampling to maintain label distribution
# _, X_train_sampled, _, y_train_sampled = train_test_split(
#     X_train, y_train, 
#     test_size=train_sample_size/len(X_train),
#     stratify=y_train,
#     random_state=42
# )

# _, X_test_sampled, _, y_test_sampled = train_test_split(
#     X_test, y_test,
#     test_size=test_sample_size/len(X_test),
#     stratify=y_test,
#     random_state=42
# )

# # Replace original data with sampled data
# X_train, y_train = X_train_sampled, y_train_sampled
# X_test, y_test = X_test_sampled, y_test_sampled


train_unique, train_counts = np.unique(y_train, return_counts=True)
train_distribution = dict(zip(train_unique, train_counts / len(y_train)))
print("Training set class distribution:")
print(train_distribution)


test_unique, test_counts = np.unique(y_test, return_counts=True)
test_distribution = dict(zip(test_unique, test_counts / len(y_test)))
print("Test set class distribution:")
print(test_distribution)

models = {
    'LogisticRegression': LogisticRegression(max_iter=2000),
    'SVM': SVC(probability=True),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(),
    'NaiveBayes': GaussianNB(),
    'GradientBoosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(eval_metric='logloss')
}

param_grids = {
    'LogisticRegression': [
        {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']  # l1只支持这两种solver
        },
        {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['elasticnet'],
            'solver': ['saga'],  # elasticnet只支持saga
            'l1_ratio': [0.2, 0.5, 0.8]
        }
    ],
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1], # 添加gamma参数
        'degree': [2, 3, 4] # poly核函数参数
    },
    'DecisionTree': {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10], # 添加分裂所需最小样本数
        'min_samples_leaf': [1, 2, 4], # 添加叶节点最小样本数
        'criterion': ['gini', 'entropy'] # 添加分裂标准选择
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None] # 添加特征选择方式
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'], # 添加权重选项
        'metric': ['euclidean', 'manhattan', 'minkowski'], # 添加距离度量
        'p': [1, 2] # Minkowski距离的参数
    },
    'NaiveBayes': {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6] # 添加平滑参数
    },
    'GradientBoosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0] # 添加样本采样比例
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0], # 特征采样
        'min_child_weight': [1, 3, 5], # 添加叶子节点最小权重
        'gamma': [0, 0.1, 0.2] # 添加分裂所需最小损失减少值
    }
}

best_models = {}

"""
GridSearchCV 会从 param_grids 中将所有可能的参数组合逐一与模型搭配，采用交叉验证（cv=3）在训练集上进行多次训练与评估。
搜索过程中会比较每种参数组合下模型的评估得分（accuracy），选出得分最高的参数配置作为最佳参数。
最终将最佳模型在测试集上进行预测与评估，输出最佳参数及相应的评估指标（准确率、混淆矩阵、ROC AUC 等）。

"""
for name, model in models.items():
    gs = GridSearchCV(model, param_grids[name], cv=4, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train, y_train)
    best_models[name] = gs.best_estimator_
    print(f"=== {name} ===")
    print("Best Params:", gs.best_params_)
    
    test_preds = gs.predict(X_test)
    probs = gs.predict_proba(X_test)[:, 1] if hasattr(gs, 'predict_proba') else None
    
    acc = accuracy_score(y_test, test_preds)
    auc_val = roc_auc_score(y_test, probs) if probs is not None else 0
    folder_name = f"{name}_acc_{acc:.4f}_auc_{auc_val:.4f}"
    folder_name = os.path.join(exp_dir, folder_name)
    os.makedirs(folder_name, exist_ok=True)
    
    # 保存模型
    joblib.dump(gs.best_estimator_, f"{folder_name}/{name}_model.joblib")
    # 保存超参
    with open(f"{folder_name}/{name}_hyperparameters.json", 'w') as f:
        json.dump(gs.best_params_, f, indent=4)
    print(classification_report(y_test, test_preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds))
    if probs is not None:
        print("ROC AUC:", auc_val)
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} ROC Curve')
        plt.legend()
        plt.savefig(f"{folder_name}/{name}_ROC.png")
        plt.close()
    print()