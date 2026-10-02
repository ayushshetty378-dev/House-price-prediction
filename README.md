# House Price Prediction Model

Mini Project 2 – Week 2 (Machine Learning & AI): Supervised Learning (Regression)

## Overview

This project predicts house sale prices using a **Linear Regression** model trained on the Kaggle *House Prices: Advanced Regression Techniques* dataset. It covers the four required tasks:

1. Train a Linear Regression model
2. Predict house prices
3. Evaluate the R² score
4. Plot predicted vs actual values

## Dataset

- Source: [Kaggle – House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
- File used: `train.csv` (1,460 houses, 81 columns including the target `SalePrice`)

## Project Structure

```
House price prediction/
├── House price prediction.py   # main script
├── train.csv                   # Kaggle dataset (download separately)
├── predicted_vs_actual.png     # output plot
└── README.md
```

## Requirements

- Python 3.9+
- pandas, numpy, scikit-learn, matplotlib

```
pip install pandas numpy scikit-learn matplotlib
```

## How to Run

1. Download `train.csv` from the Kaggle link above and place it in the project folder.
2. Run:

```
python "House price prediction.py"
```

You can also pass a custom path to the CSV:

```
python "House price prediction.py" "C:\path\to\train.csv"
```

If `train.csv` is not found, the script falls back to synthetic demo data and prints a warning. Those results are not real.

## Method

- **Cleaning:** dropped the `Id` column and any column with more than 40% missing values.
- **Numeric features:** missing values filled with the median, then standardised.
- **Categorical features:** missing values filled with the most frequent value, then one-hot encoded.
- **Model:** `LinearRegression` inside a scikit-learn `Pipeline`.
- **Split:** 80% train / 20% test (`random_state=42`).

## Results (test set)

| Metric | Value |
|--------|-------|
| R² score | 0.8852 |
| RMSE | $29,680 |
| MAE | $18,377 |
| Train R² | 0.9335 |

The model explains about 88.5% of the variation in house prices on unseen data. A typical prediction is off by about $18k. The small gap between train and test R² indicates mild overfitting.

## Output

`predicted_vs_actual.png` plots predicted prices against actual prices. Points close to the red dashed line are accurate predictions.

## Possible Improvements

- Predict `log(SalePrice)` to reduce the effect of expensive outliers
- Try Ridge or Lasso regression to reduce overfitting
- Add feature engineering (e.g. total square footage, house age)
- Compare with Random Forest
