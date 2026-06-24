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
# 固定一次人工缺失
# ==========================

np.random.seed(42)

data_with_missing = data.copy()

missing_records = pd.DataFrame()

for col in complete_cols:

    missing_idx = np.random.choice(
        data.index,
        size=int(len(data) * 0.10),
        replace=False
    )

    missing_records = pd.concat([
        missing_records,
        pd.DataFrame({
            "列名": col,
            "行索引": missing_idx,
            "真实值": data.loc[missing_idx, col]
        })
    ])

    data_with_missing.loc[
        missing_idx,
        col
    ] = np.nan

print("\n已完成人工缺失")


# ==========================
# 计算NRMSE函数
# ==========================

def calculate_mean_nrmse(imputed_df):

    nrmse_list = []

    for col in complete_cols:

        col_records = missing_records[
            missing_records["列名"] == col
        ]

        y_true = col_records["真实值"]

        y_pred = imputed_df.loc[
            col_records["行索引"],
            col
        ]

        rmse = np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )

        nrmse = rmse / (
            y_true.max() - y_true.min()
        )

        nrmse_list.append(nrmse)

    mean_nrmse = np.mean(
        nrmse_list
    )

    sd_nrmse_by_variable = np.std(
        nrmse_list,
        ddof=1
    )

    return mean_nrmse, sd_nrmse_by_variable


# ==========================
# Phase 1：筛选 n_neighbors
# ==========================

n_neighbors_list = [
    3,
    5,
    7,
    9,
    11,
    13,
    15,
    17,
    19,
    21
]

phase1_results = []

for k in n_neighbors_list:

    print(f"\nPhase 1: Testing n_neighbors={k}")

    imputer = KNNImputer(
        n_neighbors=k,
        weights="uniform"
    )

    imputed_data = imputer.fit_transform(
        data_with_missing
    )

    imputed_df = pd.DataFrame(
        imputed_data,
        columns=data.columns
    )

    # 恢复 Infection
    imputed_df["Infection"] = data["Infection"]

    mean_nrmse, sd_nrmse_by_variable = calculate_mean_nrmse(
        imputed_df
    )

    phase1_results.append({
        "n_neighbors": k,
        "Mean_NRMSE": mean_nrmse,
        "SD_NRMSE_By_Variable": sd_nrmse_by_variable
    })

phase1_df = pd.DataFrame(
    phase1_results
)

phase1_df = phase1_df.sort_values(
    "Mean_NRMSE"
)

phase1_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "KNN_Phase1_All_Results.xlsx"
    ),
    index=False
)

print("\n======================")
print("Phase 1 TOP3")
print("======================")

print(
    phase1_df.head(3)
)


# ==========================
# Phase 2：Top3 × 10 seeds
# ==========================

top3_params = phase1_df.head(3)["n_neighbors"].tolist()

phase2_results = []

for k in top3_params:

    print("\n======================")
    print(f"Phase 2: n_neighbors={k}")
    print("======================")

    run_nrmse = []

    for seed in range(100, 1100, 100):

        print(f"Seed={seed}")

        # 注意：
        # KNNImputer本身没有random_state。
        # 这里seed用于重新制造人工缺失位置。
        # 这样才能评估不同随机缺失情景下KNN的稳定性。

        np.random.seed(seed)

        data_with_missing_seed = data.copy()
        missing_records_seed = pd.DataFrame()

        for col in complete_cols:

            missing_idx = np.random.choice(
                data.index,
                size=int(len(data) * 0.10),
                replace=False
            )

            missing_records_seed = pd.concat([
                missing_records_seed,
                pd.DataFrame({
                    "列名": col,
                    "行索引": missing_idx,
                    "真实值": data.loc[missing_idx, col]
                })
            ])

            data_with_missing_seed.loc[
                missing_idx,
                col
            ] = np.nan

        # 暂时替换全局 missing_records，用于计算本次NRMSE
        old_missing_records = missing_records.copy()
        missing_records = missing_records_seed.copy()

        imputer = KNNImputer(
            n_neighbors=k,
            weights="uniform"
        )

        imputed_data = imputer.fit_transform(
            data_with_missing_seed
        )

        imputed_df = pd.DataFrame(
            imputed_data,
            columns=data.columns
        )

        imputed_df["Infection"] = data["Infection"]

        mean_nrmse, sd_nrmse_by_variable = calculate_mean_nrmse(
            imputed_df
        )

        run_nrmse.append(
            mean_nrmse
        )

        # 恢复原来的 missing_records
        missing_records = old_missing_records.copy()

    overall_mean = np.mean(
        run_nrmse
    )

    overall_sd = np.std(
        run_nrmse,
        ddof=1
    )

    ci_low = np.percentile(
        run_nrmse,
        2.5
    )

    ci_high = np.percentile(
        run_nrmse,
        97.5
    )

    phase2_results.append({
        "n_neighbors": k,
        "Mean_NRMSE": overall_mean,
        "SD_NRMSE": overall_sd,
        "CI_Low": ci_low,
        "CI_High": ci_high
    })


phase2_df = pd.DataFrame(
    phase2_results
)

phase2_df = phase2_df.sort_values(
    "Mean_NRMSE"
)

phase2_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "KNN_Phase2_Final_Ranking.xlsx"
    ),
    index=False
)

print("\n======================")
print("KNN FINAL RESULT")
print("======================")

print(
    phase2_df
)

best = phase2_df.iloc[0]

print("\n最佳KNN参数：")
print(f"n_neighbors = {best['n_neighbors']}")
print(f"Mean NRMSE = {best['Mean_NRMSE']:.5f}")
print(f"SD NRMSE = {best['SD_NRMSE']:.5f}")
print(f"95% CI = [{best['CI_Low']:.5f}, {best['CI_High']:.5f}]")

print("\n完成")