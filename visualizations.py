"""
visualizations.py
Generate all visualizations for the Housing Price Prediction Model dataset.
Handles:
- Data loading & sampling for efficiency
- Cleaning & preprocessing
- EDA plots (distribution, heatmap, relationships)
- Training visualizations: Loss vs Epoch, AUPRC vs Epoch
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc


# ============================================================
# 1. LOAD DATA (Efficient for 2.2M rows)
# ============================================================
def load_data(filepath="data.txt"):
    use_cols = [
        "price", "bed", "bath", "acre_lot", "city", "state",
        "zip_code", "house_size"
    ]
    
    print("Loading dataset...")
    # df = pd.read_csv(filepath, usecols=use_cols)
    df = pd.read_csv("realtor-data.zip.csv", usecols=use_cols, low_memory=False)


    print(f"Total rows loaded: {len(df):,}")
    return df


# ============================================================
# 2. CLEAN & SAMPLE DATA
# ============================================================
def preprocess(df, sample_size=100000):
    print("Cleaning data...")
    
    df = df.dropna(subset=["price", "bed", "bath", "house_size"])
    
    df = df[
        (df["price"] > 1000) &
        (df["bed"] >= 1) & (df["bed"] <= 10) &
        (df["bath"] >= 1) & (df["bath"] <= 10) &
        (df["house_size"] > 300) & (df["house_size"] < 10000)
    ]

    print("Sampling dataset for plotting...")
    df_sample = df.sample(sample_size, random_state=42)

    print(f"Rows after cleaning & sampling: {len(df_sample):,}")
    return df_sample


# ============================================================
# 3. PLOT: Log Price Distribution
# ============================================================
def plot_price_distribution(df):
    plt.figure(figsize=(8,5))
    sns.histplot(np.log1p(df["price"]), kde=True)
    plt.title("Log-Transformed Price Distribution")
    plt.xlabel("log(price)")
    plt.savefig("plots/price_distribution.png")
    plt.show()


# ============================================================
# 4. PLOT: Scatter (House Size vs Price)
# ============================================================
def plot_size_vs_price(df):
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df["house_size"], y=df["price"], alpha=0.3)
    plt.title("Price vs House Size")
    plt.xlabel("House Size (sq ft)")
    plt.ylabel("Price ($)")
    plt.yscale("log")
    plt.savefig("plots/size_vs_price.png")
    plt.show()


# ============================================================
# 5. PLOT: Price vs Bedrooms/Bathrooms
# ============================================================
def plot_bed_bath_price(df):
    # Bedrooms
    plt.figure(figsize=(8,5))
    sns.boxplot(x=df["bed"], y=np.log1p(df["price"]))
    plt.title("Price vs Bedroom Count")
    plt.savefig("plots/price_vs_bedrooms.png")
    plt.show()

    # Bathrooms
    plt.figure(figsize=(8,5))
    sns.boxplot(x=df["bath"], y=np.log1p(df["price"]))
    plt.title("Price vs Bathroom Count")
    plt.savefig("plots/price_vs_bathrooms.png")
    plt.show()


# ============================================================
# 6. PLOT: Correlation Heatmap
# ============================================================
def plot_correlation_heatmap(df):
    corr = df[["price", "bed", "bath", "acre_lot", "house_size"]].corr()
    
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Matrix")
    plt.savefig("plots/correlation_heatmap.png")
    plt.show()


# ============================================================
# 7. PLOT: Average Price by State (Top 15)
# ============================================================
def plot_price_by_state(df):
    top_states = df.groupby("state")["price"].mean().sort_values(ascending=False).head(15)

    plt.figure(figsize=(12,5))
    sns.barplot(x=top_states.index, y=top_states.values)
    plt.title("Average Price by State (Top 15)")
    plt.xticks(rotation=45)
    plt.yscale("log")
    plt.ylabel("Average Price ($)")
    plt.savefig("plots/price_by_state.png")
    plt.show()


# ============================================================
# 8. TRAINING VISUALIZATION — Loss vs Epoch
# ============================================================
def plot_loss_curve(train_losses, val_losses=None):
    plt.figure(figsize=(8,5))

    plt.plot(train_losses, label="Train Loss", linewidth=2)

    if val_losses:
        plt.plot(val_losses, label="Val Loss", linewidth=2)

    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plots/loss_curve.png")
    plt.show()


# ============================================================
# 9. TRAINING VISUALIZATION — AUPRC vs Epoch
# ============================================================
def plot_auprc_curve(y_true_list, y_pred_list):
    """
    y_true_list: list of arrays (per epoch)
    y_pred_list: list of arrays (per epoch)
    """

    auprc_scores = []

    for y_true, y_pred in zip(y_true_list, y_pred_list):
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        auprc = auc(recall, precision)
        auprc_scores.append(auprc)

    plt.figure(figsize=(8,5))
    plt.plot(auprc_scores, label="AUPRC per Epoch", linewidth=2)
    plt.title("AUPRC vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("AUPRC")
    plt.savefig("plots/auprc_curve.png")
    plt.show()


# ============================================================
# MAIN RUNNER
# ============================================================
if __name__ == "__main__":
    import os

    # Create output folder
    os.makedirs("plots", exist_ok=True)

    df = load_data()
    df_sample = preprocess(df)

    plot_price_distribution(df_sample)
    plot_size_vs_price(df_sample)
    plot_bed_bath_price(df_sample)
    plot_correlation_heatmap(df_sample)
    plot_price_by_state(df_sample)

    print("\nAll visualizations saved in /plots folder.")
