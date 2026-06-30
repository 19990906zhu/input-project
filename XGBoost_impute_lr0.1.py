import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

# ==========================
# 1. 加载数据（只需修改这里）
# ==========================
data_path = r"D:\萝卜投稿专用\Original Research - Spinal Infection Following Internal Fixation Surgery\插值选择\外部验证集_原始数据.xlsx"
data = pd.read_excel(data_path)

# ==========================
# 2. Infection 只作为结局标签，不作为插补预测因子
# ==========================
if "Infection" not in data.columns:
    raise ValueError("数据中必须包含 'Infection' 列作为结局标签")

infection_label = data["Infection"].copy()
feature_cols = [col for col in data.columns if col != "Infection"]
feature_data = data[feature_cols].copy()

# 找出所有完整列：只在特征列中寻找，Infection 不参与
complete_cols = [col for col in feature_cols if feature_data[col].isnull().sum() == 0]
print("完整特征列有:", complete_cols)

if not complete_cols:
    raise ValueError("没有找到完整特征列！")

# ==========================
# 3. XGBoost 插补：只对特征矩阵 feature_data 插补
# ==========================
imputer = IterativeImputer(
    estimator=XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    ),
    max_iter=12,
    random_state=42,
)

imputed_feature_data = imputer.fit_transform(feature_data)

imputed_feature_df = pd.DataFrame(
    imputed_feature_data,
    columns=feature_cols,
    index=data.index,
)

# ==========================
# 4. 拼回原始 Infection 标签，保持原始列顺序
# ==========================
imputed_df = imputed_feature_df.copy()
imputed_df["Infection"] = infection_label.values
imputed_df = imputed_df[data.columns]

# ==========================
# 5. 导出最终插补表
# ==========================
output_path = r"D:\萝卜投稿专用\Original Research - Spinal Infection Following Internal Fixation Surgery\插值选择\Xgboost\lr=0.1\外部验证集的处理\XGBoost_Imputed_lr0.1_external_test.xlsx"
imputed_df.to_excel(output_path, index=False)

print("完成：Infection 未作为预测因子参与插补，只在最后原样拼回。")
print("输出文件:", output_path)
