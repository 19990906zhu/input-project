import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 加载数据
file_path = r"D:\萝卜投稿专用\Original Research - Spinal Infection Following Internal Fixation Surgery\插值选择\Xgboost\lr=0.1\P小于0.05_按临床动态标准共线性删除后.xlsx"
df = pd.read_excel(file_path)

# 选择需要标准化的列（排除标签列）
columns_to_standardize = [
    c for c in df.columns
    if c != "Infection"
]

# 初始化标准化器并保存参数
scaler = StandardScaler()
scaler.fit(df[columns_to_standardize])  # 仅计算μ和σ，不转换数据

# 查看参数（可选）
print("均值(μ):", scaler.mean_)
print("标准差(σ):", scaler.scale_)

# 转换训练集数据
df[columns_to_standardize] = scaler.transform(df[columns_to_standardize])

# 保存标准化器和参数（供验证集使用）
import joblib
joblib.dump(scaler, 'standard_scaler.pkl')

# 保存标准化后的数据
df.to_excel(r"D:\萝卜投稿专用\Original Research - Spinal Infection Following Internal Fixation Surgery\插值选择\Xgboost\lr=0.1\训练集-z-scores.xlsx", index=False)
