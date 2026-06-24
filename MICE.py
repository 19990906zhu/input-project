import numpy as np
import pandas as pd
import os

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# ==========================
# 配置区域
# ==========================

INPUT_FILE = r"D:\萝卜投稿专用\Original Research - Spinal Infection Following Internal Fixation Surgery\图表及其来源\KNN与MICE的选择，归一化\插值选择\训练集_原始数据.xlsx"

OUTPUT_DIR = r"D:\萝卜投稿专用\Original Research - Spinal Infection Following Internal Fixation Surgery\图表及其来源\KNN与MICE的选择，归一化\插值选择\MICE"

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

# ==========================
# 制造10%缺失
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
# 开始10次MICE
# ==========================

n_estimators_list = [
    10,
    20,
    30,
    50,
    80,
    100
]

phase1_results = []

for n_est in n_estimators_list:

    print(
        f"\nTesting n_estimators={n_est}"
    )

    imputer = IterativeImputer(

        estimator=RandomForestRegressor(

            n_estimators=n_est,

            random_state=42,

            n_jobs=-1
        ),

        max_iter=15,

        random_state=42
    )

    imputed_data = imputer.fit_transform(
        data_with_missing
    )

    imputed_df = pd.DataFrame(
        imputed_data,
        columns=data.columns
    )

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

    phase1_results.append({

        "n_estimators": n_est,

        "Mean_NRMSE": np.mean(
            nrmse_list
        )
    })

phase1_df = pd.DataFrame(
    phase1_results
)

phase1_df = phase1_df.sort_values(
    "Mean_NRMSE"
)

print("\nTOP3")

print(
    phase1_df.head(3)
)


# =====================================
# Phase 2
# =====================================

top3_params = [
    10,
    20,
    30
]

phase2_results = []

for n_est in top3_params:

    print("\n===================")
    print(
        f"Testing "
        f"n_estimators={n_est}"
    )
    print("===================")

    run_nrmse = []

    for seed in range(100, 1100, 100):

        print(f"Seed={seed}")

        imputer = IterativeImputer(

            estimator=RandomForestRegressor(

                n_estimators=n_est,

                random_state=seed,

                n_jobs=-1
            ),

            max_iter=15,

            random_state=seed
        )

        imputed_data = imputer.fit_transform(
            data_with_missing
        )

        imputed_df = pd.DataFrame(
            imputed_data,
            columns=data.columns
        )

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

        run_nrmse.append(
            mean_nrmse
        )

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

        "n_estimators": n_est,

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
    "MICE_RF_Phase2_Final_Ranking.xlsx",
    index=False
)

print("\n===================")
print("FINAL RESULT")
print("===================")

print(
    phase2_df
)