import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import optuna
import matplotlib.pyplot as plt
import joblib
import os
import requests

# Constants
LOOKBACK_HOURS = 4320  # 6 months
BASE_URL = "https://cryptocurrency.azurewebsites.net/api"
HEADERS = {"Accept": "application/json"}
RANDOM_STATE = 42

# Fetch data from API
df_coin = pd.DataFrame(requests.get(f"{BASE_URL}/CoinData?lookbackHours={LOOKBACK_HOURS}", headers=HEADERS).json())
df_investor = pd.DataFrame(requests.get(f"{BASE_URL}/InvestorGrade?lookbackHours={LOOKBACK_HOURS}", headers=HEADERS).json())
df_sentiment = pd.DataFrame(requests.get(f"{BASE_URL}/Sentiment?lookbackHours={LOOKBACK_HOURS}", headers=HEADERS).json())

# Convert date columns to datetime
df_coin["date"] = pd.to_datetime(df_coin["date"])
df_investor["date"] = pd.to_datetime(df_investor["date"])
df_sentiment["date"] = pd.to_datetime(df_sentiment["date"])

# Floor InvestorGrade date to day
df_investor["date"] = df_investor["date"].dt.floor("D")

# Preserve hourly date for coin and sentiment, but also floor to join with investor
df_coin["dateHour"] = df_coin["date"]
df_sentiment["dateHour"] = df_sentiment["date"]
df_coin["date"] = df_coin["date"].dt.floor("D")
df_sentiment["date"] = df_sentiment["date"].dt.floor("D")

# Merge coin + investor on Symbol + floored date
df = df_coin.merge(df_investor, on=["symbol", "date"], how="left")

# Merge in sentiment on Symbol + exact hour
df = df.merge(df_sentiment, left_on=["symbol", "dateHour"], right_on=["symbol", "dateHour"], how="inner")

# Use full datetime again
df["date"] = df["dateHour"]
df.drop(columns=["dateHour"], inplace=True)

# Sort and shift the target for next-hour prediction
df.sort_values(by=["symbol", "date"], inplace=True)
df["percentChange1h"] = df.groupby("symbol")["percentChange1h"].shift(-1)

# Feature engineering
df["sentimentScore"] = df["positiveReddit"] - df["negativeReddit"]
# Lag Features
for col in ["price", "volume24h", "volumeChange24h", "fearandGreed", "sentimentScore"]:
    df[f"{col}_lag_1"] = df.groupby("symbol")[col].shift(1)
    df[f"{col}_lag_2"] = df.groupby("symbol")[col].shift(2)
# Rolling Features
df["price_rolling_mean_3h"] = df.groupby("symbol")["price"].transform(lambda x: x.rolling(3).mean())
df["volume_rolling_std_6h"] = df.groupby("symbol")["volume24h"].transform(lambda x: x.rolling(6).std())
df["sentiment_rolling_mean_3h"] = df.groupby("symbol")["sentimentScore"].transform(lambda x: x.rolling(3).mean())
#Momentum Features
df["price_momentum_1h"] = df["price"] - df["price_lag_1"]
df["sentiment_momentum_1h"] = df["sentimentScore"] - df["sentimentScore_lag_1"]
# Encode Text
le = LabelEncoder()
df["symbol_encoded"] = le.fit_transform(df["symbol"])

# Drop rows with any missing values
df.dropna(inplace=True)

# Feature and target selection
feature_cols = [
    "symbol_encoded", "fearandGreed", "price", "volume24h", "volumeChange24h",
    "percentChange1h", "percentChange24h", "percentChange7d", "percentChange30d",
    "percentChange60d", "percentChange90d", "sentimentScore", "postCountReddit",
    "tradingSignal", "tokenTrend", "tradingSignalsReturns", "holdingReturns",
    "tmTraderGrade", "tmInvestorGrade", "taGrade", "quantGrade", "tmTraderGrade24hPctChange",

    # 🔁 Lag features (1h and 2h)
    "price_lag_1", "price_lag_2",
    "volume24h_lag_1", "volume24h_lag_2",
    "volumeChange24h_lag_1", "volumeChange24h_lag_2",
    "fearandGreed_lag_1", "fearandGreed_lag_2",
    "sentimentScore_lag_1", "sentimentScore_lag_2",

    # 📊 Rolling stats
    "price_rolling_mean_3h", "volume_rolling_std_6h", "sentiment_rolling_mean_3h",

    # 📈 Momentum features
    "price_momentum_1h", "sentiment_momentum_1h"
]
X = df[feature_cols]
y = df["percentChange1h"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Optuna optimization
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
    }
    model = xgb.XGBRegressor(**params, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds)

# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
best_params = study.best_params

# Final model
final_model = xgb.XGBRegressor(**study.best_params, random_state=RANDOM_STATE)
final_model.fit(X_train, y_train)

# Get Feature Importance
model = xgb.XGBRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Evaluate top-N features (from 5 to 30)
results = []
top_feature_sets = {}

for n in range(5, min(31, len(importance) + 1)):
    top_n_features = importance.head(n).index.tolist()
    X_top = X[top_n_features]
    X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(X_top, y, shuffle=False, test_size=0.2)
    model_n = xgb.XGBRegressor(**best_params, random_state=42)
    model_n.fit(X_train_top, y_train_top)
    y_pred = model_n.predict(X_test_top)
    mae = mean_absolute_error(y_test_top, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_top, y_pred))
    results.append((n, mae, rmse))
    top_feature_sets[n] = top_n_features
    print(f"Top {n} features — MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Save evaluation table
results_df = pd.DataFrame(results, columns=["TopN", "MAE", "RMSE"])
os.makedirs("models", exist_ok=True)
results_df.to_csv("models/1h_prediction_Stable_topN_evaluation_results.csv", index=False)

# Select best N based on lowest MAE
best_n = results_df.loc[results_df["MAE"].idxmin(), "TopN"]
best_features = top_feature_sets[best_n]

# Retrain final model with best-N features on full dataset
X_final = X[best_features]
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_final, y, shuffle=False, test_size=0.2)
final_model = xgb.XGBRegressor(**best_params, random_state=42)
final_model.fit(X_train_final, y_train_final)

# Evaluate final model
y_pred_final = final_model.predict(X_test_final)
mae = mean_absolute_error(y_test_final, y_pred_final)
rmse = np.sqrt(mean_squared_error(y_test_final, y_pred_final))

# Save final model and feature list
joblib.dump(final_model, "models/1h_prediction_Stable_xgb_regression_model.pkl")
pd.Series(best_features).to_csv("models/1h_prediction_Stable_final_used_features.csv", index=False)
joblib.dump(le, "models/1h_prediction_symbol_label_encoder.pkl")

# Save predicted vs actual plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.scatter(y_test_final, y_pred_final, alpha=0.3)
plt.xlabel("Actual % Change")
plt.ylabel("Predicted % Change")
plt.title(f"Predicted vs Actual\\nMAE: {mae:.4f}, RMSE: {rmse:.4f}")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/1h_prediction_Stable_predicted_vs_actual.png")

# Save feature importances
feature_importance = pd.Series(final_model.feature_importances_, index=X_final.columns)
feature_importance.sort_values(ascending=False).to_csv("models/1h_prediction_Stable_feature_importance.csv")

# Output
print("Best Parameters:", best_params)
print(f"✅ Final model trained using Top {int(best_n)} features")
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")