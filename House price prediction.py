
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "SalePrice"
OUT_DIR = Path(__file__).parent


def make_synthetic(n=1460, seed=42):
    """Stand-in data with the same column names as the Kaggle set (for demo only)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "OverallQual": rng.integers(1, 11, n),
        "GrLivArea": rng.normal(1500, 500, n).clip(400, 5000),
        "GarageCars": rng.integers(0, 4, n),
        "TotalBsmtSF": rng.normal(1000, 400, n).clip(0, 3000),
        "YearBuilt": rng.integers(1900, 2010, n),
        "FullBath": rng.integers(0, 4, n),
        "LotArea": rng.normal(10000, 3500, n).clip(1500, 50000),
        "Neighborhood": rng.choice(["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst"], n),
        "KitchenQual": rng.choice(["Ex", "Gd", "TA", "Fa"], n, p=[.1, .35, .5, .05]),
    })
    nb = {"NAmes": 0, "CollgCr": 15000, "OldTown": -10000, "Edwards": -5000, "Somerst": 30000}
    kq = {"Ex": 40000, "Gd": 15000, "TA": 0, "Fa": -15000}
    df[TARGET] = (
        15000 * df.OverallQual + 55 * df.GrLivArea + 9000 * df.GarageCars
        + 20 * df.TotalBsmtSF + 250 * (df.YearBuilt - 1900) + 5000 * df.FullBath
        + 0.8 * df.LotArea + df.Neighborhood.map(nb) + df.KitchenQual.map(kq)
        - 60000 + rng.normal(0, 18000, n)
    ).clip(30000)
    df.loc[rng.choice(n, 40, replace=False), "TotalBsmtSF"] = np.nan  # some missing values
    return df


def load_data():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR / "train.csv"
    if path.exists():
        print(f"Loaded Kaggle data: {path}")
        return pd.read_csv(path), True
    print("train.csv not found -> using SYNTHETIC demo data (results are not real Kaggle results).")
    return make_synthetic(), False


def main():
    df, real = load_data()
    print(f"Shape: {df.shape}")

    df = df.drop(columns=[c for c in ["Id"] if c in df.columns])
    df = df.loc[:, df.isna().mean() < 0.4]

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

    model = Pipeline([("pre", pre), ("lr", LinearRegression())])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    
    model.fit(X_train, y_train)

    
    pred = model.predict(X_test)


    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    print("\n--- Test set results ---")
    print(f"R2 score : {r2:.4f}")
    print(f"RMSE     : {rmse:,.0f}")
    print(f"MAE      : {mae:,.0f}")
    print(f"R2 (train): {model.score(X_train, y_train):.4f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, pred, alpha=0.6, edgecolor="k", linewidth=0.3)
    lo, hi = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", label="Perfect prediction")
    ax.set_xlabel("Actual SalePrice")
    ax.set_ylabel("Predicted SalePrice")
    ax.set_title(f"Predicted vs Actual House Prices (R² = {r2:.3f})"
                 + ("" if real else "\n[synthetic demo data]"))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "predicted_vs_actual.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved plot -> {out}")

    sample = pd.DataFrame({"Actual": y_test.values[:5].round(0),
                           "Predicted": pred[:5].round(0)})
    print("\nSample predictions:\n", sample.to_string(index=False))


if __name__ == "__main__":
    main()