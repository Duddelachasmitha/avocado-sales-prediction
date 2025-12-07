
# train_model.py
# Uses Avocado.csv to train a model that predicts "Total Volume"
# based on AveragePrice, date (year/month), type, and region.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Load dataset
df = pd.read_csv("Avocado.csv")

# 2. Basic cleaning
# Drop the unnecessary index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Convert Date to datetime and extract month
df["Date"] = pd.to_datetime(df["Date"])
df["month"] = df["Date"].dt.month
# 'year' column already exists in this dataset

# 3. Define features and target
feature_cols = ["AveragePrice", "year", "month", "type", "region"]
target_col = "Total Volume"

X = df[feature_cols]
y = df[target_col]

numeric_features = ["AveragePrice", "year", "month"]
categorical_features = ["type", "region"]

# 4. Preprocessing
numeric_transformer = "passthrough"
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 5. Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ]
)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train
pipe.fit(X_train, y_train)

# 8. Evaluate
y_pred = pipe.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:,.2f}")
print(f"R²:  {r2:.3f}")

# 9. Save model and feature metadata
joblib.dump(pipe, "avocado_model.pkl")
joblib.dump(feature_cols, "avocado_feature_cols.pkl")

print("Model saved as avocado_model.pkl")
