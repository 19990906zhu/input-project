
import pandas as pd
import numpy as np

from scipy.stats import (
    shapiro,
    ttest_ind,
    mannwhitneyu
)

# =========================
# 配置
# =========================

INPUT_FILE = r"D:\萝卜投稿专用\Original Research - Spinal Infection Following Internal Fixation Surgery\插值选择\Xgboost\lr=0.1\训练集_lr0.1.xlsx"
OUTPUT_FILE = r"D:\萝卜投稿专用\Original Research - Spinal Infection Following Internal Fixation Surgery\插值选择\Xgboost\lr=0.1\两组比较_正态性_t或U检验.xlsx"

TARGET_COL = "Infection"

# =========================
# 读取数据
# =========================

df = pd.read_excel(INPUT_FILE)

if TARGET_COL not in df.columns:
    raise ValueError(f"找不到结局列: {TARGET_COL}")

group0 = df[df[TARGET_COL] == 0]
group1 = df[df[TARGET_COL] == 1]

results = []

# =========================
# 遍历所有变量
# =========================

for col in df.columns:

    if col == TARGET_COL:
        continue

    # 只分析数值变量
    if not pd.api.types.is_numeric_dtype(df[col]):
        continue

    x0 = group0[col].dropna()
    x1 = group1[col].dropna()

    # 样本太少直接跳过
    if len(x0) < 3 or len(x1) < 3:
        continue

    # ---------------------
    # 正态性检验
    # ---------------------

    try:
        p_norm0 = shapiro(x0).pvalue
    except:
        p_norm0 = np.nan

    try:
        p_norm1 = shapiro(x1).pvalue
    except:
        p_norm1 = np.nan

    # ---------------------
    # 选择检验方法
    # ---------------------

    if (
        not np.isnan(p_norm0)
        and not np.isnan(p_norm1)
        and p_norm0 > 0.05
        and p_norm1 > 0.05
    ):
        test_name = "Welch_t_test"

        stat, p_value = ttest_ind(
            x0,
            x1,
            equal_var=False
        )

    else:

        test_name = "Mann_Whitney_U"

        stat, p_value = mannwhitneyu(
            x0,
            x1,
            alternative="two-sided"
        )

    results.append({
        "Variable": col,

        "Group0_N": len(x0),
        "Group1_N": len(x1),

        "Group0_Mean": x0.mean(),
        "Group0_SD": x0.std(),

        "Group1_Mean": x1.mean(),
        "Group1_SD": x1.std(),

        "Normality_P_Group0": p_norm0,
        "Normality_P_Group1": p_norm1,

        "Test": test_name,

        "Statistic": stat,
        "P_Value": p_value
    })

# =========================
# 输出
# =========================

result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    "P_Value",
    ascending=True
)

result_df.to_excel(
    OUTPUT_FILE,
    index=False
)

print(f"完成！结果保存至：{OUTPUT_FILE}")
