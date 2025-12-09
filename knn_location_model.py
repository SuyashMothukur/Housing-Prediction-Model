import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("housing_data_cleaned.csv")

target = "price"

exclude_cols = [
    target,
    "log_price",
    "city",
    "state",
    "region"
]

feature_cols = [c for c in df.columns if c not in exclude_cols]


if len(df) > 10000:
    print(f"Dataset has {len(df)} rows. Sampling for speed...")
    df = df.sample(n=5000, random_state=42)   # preserves distribution
    print("Sampled 5,000 rows for fast KNN computation.")

X = df[feature_cols].values
y = df[target].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_values = [1, 3, 5, 7, 10, 15]   # these always capture the shape of KNN performance
rmse_values = []

for k in k_values:
    knn = KNeighborsRegressor(
        n_neighbors=k,
        algorithm="kd_tree",   # optimizes search
        weights="distance"     # improves accuracy
    )
    knn.fit(X_train_scaled, y_train)
    pred = knn.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    rmse_values.append(rmse)

# Plot validation curve
plt.figure(figsize=(8,5))
plt.plot(k_values, rmse_values, marker='o')
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("RMSE")
plt.title("Fast KNN Validation Curve")
plt.grid(True)
plt.savefig("knn_fast_validation_curve.png", dpi=300)
plt.show()

# Select best k
best_k = k_values[np.argmin(rmse_values)]
print(f"Best k: {best_k}")

knn = KNeighborsRegressor(n_neighbors=best_k, algorithm="kd_tree", weights="distance")
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nFAST KNN Performance:")
print(f"MAE:  {mae:,.0f}")
print(f"RMSE: {rmse:,.0f}")
print(f"R²:   {r2:.4f}")

# Scatter plot: actual vs predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"KNN Predictions vs Actual (k={best_k})")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.savefig("knn_fast_predictions_vs_actual.png", dpi=300)
plt.show()

# Error distribution
errors = y_test - y_pred

plt.figure(figsize=(8,5))
plt.hist(errors, bins=50, alpha=0.7)
plt.title("KNN Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.grid(True)
plt.savefig("knn_fast_error_distribution.png", dpi=300)
plt.show()
