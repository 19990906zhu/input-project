# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = r"D:\萝卜投稿专用\Original Research - Spinal Infection Following Internal Fixation Surgery\插值选择\Xgboost\lr=0.1\训练集_lr0.1_P小于0.05筛选后.xlsx"
OUTPUT_DIR = r"D:\萝卜投稿专用\Original Research - Spinal Infection Following Internal Fixation Surgery\插值选择\Xgboost\lr=0.1"
TARGET_COLS = ["Infection"]
CORR_METHOD = "pearson"

FIG_WIDTH = 18
FIG_HEIGHT = 16
DPI = 600
SHOW_VALUES = True
VALUE_FONT_SIZE = 5
LABEL_FONT_SIZE = 8
TITLE_FONT_SIZE = 16

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_excel(INPUT_FILE)

    drop_cols = [c for c in TARGET_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.dropna(axis=1, how="all")

    if numeric_df.isna().sum().sum() > 0:
        numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

    corr = numeric_df.corr(method=CORR_METHOD)
    corr.to_excel(os.path.join(OUTPUT_DIR, "Pearson_correlation_matrix.xlsx"))

    n = corr.shape[0]
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax = plt.gca()

    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=LABEL_FONT_SIZE)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=LABEL_FONT_SIZE)
    ax.set_yticklabels(corr.index, fontsize=LABEL_FONT_SIZE)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    if SHOW_VALUES:
        for i in range(n):
            for j in range(n):
                val = corr.iloc[i, j]
                text_color = "white" if abs(val) > 0.60 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=VALUE_FONT_SIZE, color=text_color)

    ax.set_title(f"{CORR_METHOD.capitalize()} Correlation Matrix",
                 fontsize=TITLE_FONT_SIZE, pad=20)

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, r"D:\萝卜投稿专用\Original Research - Spinal Infection Following Internal Fixation Surgery\插值选择\Xgboost\lr=0.1\heatmap_1.jpg")
   
    plt.savefig(png_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    print("Finished")
    print("JPG:", png_path)


if __name__ == "__main__":
    main()
