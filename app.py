
# app.py

from flask import Flask, render_template, request
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load trained model and feature list
model = joblib.load("avocado_model.pkl")
feature_cols = joblib.load("avocado_feature_cols.pkl")

# Load dataset once to get list of regions
df = pd.read_csv("Avocado.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

regions = sorted(df["region"].unique())
types = sorted(df["type"].unique())  # usually ['conventional', 'organic']

@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        prediction=None,
        regions=regions,
        types=types,
        error=None
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        date_str = request.form.get("date")
        avg_price_str = request.form.get("average_price")
        avo_type = request.form.get("type")
        region = request.form.get("region")

        if not date_str or not avg_price_str or not avo_type or not region:
            raise ValueError("Please fill in all fields.")

        avg_price = float(avg_price_str)

        # Convert date to year and month
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        year = dt.year
        month = dt.month

        # Build a one-row DataFrame with same columns as training
        data_dict = {
            "AveragePrice": avg_price,
            "year": year,
            "month": month,
            "type": avo_type,
            "region": region,
        }

        new_data = pd.DataFrame([data_dict], columns=feature_cols)

        # Predict
        pred_volume = model.predict(new_data)[0]

        return render_template(
            "index.html",
            prediction=round(pred_volume),
            date=date_str,
            avg_price=avg_price,
            avo_type=avo_type,
            region=region,
            regions=regions,
            types=types,
            error=None
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=None,
            regions=regions,
            types=types,
            error=str(e)
        )

if __name__ == "__main__":
    app.run(debug=True)
