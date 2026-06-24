import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# 1. 加载数据（只需修改这里↓↓↓）
data_path = "D:\萝卜投稿专用\Original Research - Spinal Infection Following Internal Fixation Surgery\图表及其来源\KNN与MICE的选择，归一化\插值选择\训练集_原始数据.xlsx"  
data = pd.read_excel(data_path)

# 确保数据中包含Infection列
if 'Infection' not in data.columns:
    raise ValueError("数据中必须包含'Infection'列作为标签")

# 2. 找出所有完整列（无缺失值的列），不包括Infection列
complete_cols = [col for col in data.columns if (data[col].isnull().sum() == 0) and (col != 'Infection')]
print("完整列有:", complete_cols)

if not complete_cols:
    raise ValueError("没有找到完整列！")

# 3. 对每个完整列随机删除10%数据（制造缺失）
np.random.seed(42)  # 固定随机种子
data_with_missing = data.copy()
missing_records = pd.DataFrame()  # 记录被删除的真实值

for col in complete_cols:
    missing_idx = np.random.choice(
        data.index,
        size=int(len(data)*0.1),
        replace=False
    )
    # 记录被删除的值
    missing_records = pd.concat([
        missing_records,
        pd.DataFrame({
            '列名': col,
            '行索引': missing_idx,
            '真实值': data.loc[missing_idx, col]
        })
    ])
    # 设置缺失值
    data_with_missing.loc[missing_idx, col] = np.nan

# 4. XGBoost插值（包含Infection作为预测因子）
# ==========================
# XGBoost插补
# ==========================

imputer = IterativeImputer(

    estimator=XGBRegressor(

        n_estimators=200,

        learning_rate=0.07,

        random_state=42,

        n_jobs=-1

    ),

    max_iter=12,

    random_state=42
)

imputed_data = imputer.fit_transform(
    data
)

imputed_df = pd.DataFrame(
    imputed_data,
    columns=data.columns
)

# 恢复 Infection
imputed_df["Infection"] = data["Infection"]

# ==========================
# 导出最终插补表
# ==========================

imputed_df.to_excel(

    r"D:\萝卜投稿专用\Original Research - Spinal Infection Following Internal Fixation Surgery\图表及其来源\KNN与MICE的选择，归一化\插值选择\XGBoost_Imputed.xlsx",

    index=False
)

print("完成")
