# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os

from sklearn.impute import KNNImputer


# ==========================
# 配置区域
# ==========================

INPUT_FILE = r"D:\萝卜投稿专用\Original Research - Spinal Infection Following Internal Fixation Surgery\图表及其来源\KNN与MICE的选择，归一化\插值选择\训练集_原始数据.xlsx"

OUTPUT_DIR = r"D:\萝卜投稿专用\Original Research - Spinal Infection Following Internal Fixation Surgery\图表及其来源\KNN与MICE的选择，归一化\插值选择\KNN"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================
# 读取数据
# ==========================

print("读取数据...")

data = pd.read_excel(INPUT_FILE)

if "Infection" not in data.columns:
    raise ValueError("缺少 Infection 列")


# ==========================
# 找完整列
# ==========================

complete_cols = [
    col for col in data.columns
    if data[col].isnull().sum() == 0
    and col != "Infection"
]

print("\n完整列:")
print(complete_cols)

if not complete_cols:
    raise ValueError("没有找到完整列，无法计算NRMSE")


# ==========================
# KNN 插补
# ==========================

imputer = KNNImputer(

    n_neighbors=7,

    weights="uniform"
)

print("开始插补...")

imputed_data = imputer.fit_transform(
    data
)

imputed_df = pd.DataFrame(
    imputed_data,
    columns=data.columns
)

imputed_df["Infection"] = data["Infection"]

# ==========================
# 导出结果
# ==========================

save_path = os.path.join(
    OUTPUT_DIR,
    "KNN_k7.xlsx"
)

imputed_df.to_excel(
    save_path,
    index=False
)

print("完成")
print(save_path)