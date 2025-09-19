from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from models import db, FinanceRecord
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import io
import os

app = Flask(__name__, template_folder="templates")
CORS(app)

# --- Database setup ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///finance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    db.create_all()

# --- Forecast helpers (Linear + RandomForest) ---
def build_df_from_records(records):
    df = pd.DataFrame([r.to_dict() for r in records])
    # ensure chronological order if months are string names -> keep insertion order
    return df.reset_index(drop=True)

def forecast_models(df, months=6):
    """
    Returns a dict with forecast DataFrames for:
      - linear (sklearn LinearRegression)
      - rf (RandomForestRegressor)
      - combined (average of the two)
    Each forecast DF has columns: month, revenue, expenses, profit
    """
    if df.empty:
        empty = pd.DataFrame(columns=["month", "revenue", "expenses", "profit"])
        return {"linear": empty, "rf": empty, "combined": empty}

    # Use numeric time index as feature; for small datasets this is OK
    n = len(df)
    X = np.arange(n).reshape(-1, 1)

    future_X = np.arange(n, n + months).reshape(-1, 1)
    months_labels = [f"Future-{i+1}" for i in range(months)]

    forecasts = {}
    # Linear models
    lr_rev = LinearRegression().fit(X, df["revenue"])
    lr_exp = LinearRegression().fit(X, df["expenses"])
    lr_rev_pred = lr_rev.predict(future_X)
    lr_exp_pred = lr_exp.predict(future_X)
    lr_df = pd.DataFrame({
        "month": months_labels,
        "revenue": lr_rev_pred,
        "expenses": lr_exp_pred
    })
    lr_df["profit"] = lr_df["revenue"] - lr_df["expenses"]
    forecasts["linear"] = lr_df

    # Random Forest: use same X; n_estimators small for speed
    # For very small datasets RF may overfit — it's okay for demo/hackathon
    try:
        rf_rev = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df["revenue"])
        rf_exp = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df["expenses"])
        rf_rev_pred = rf_rev.predict(future_X)
        rf_exp_pred = rf_exp.predict(future_X)
        rf_df = pd.DataFrame({
            "month": months_labels,
            "revenue": rf_rev_pred,
            "expenses": rf_exp_pred
        })
        rf_df["profit"] = rf_df["revenue"] - rf_df["expenses"]
        forecasts["rf"] = rf_df
    except Exception:
        # fallback to linear if RF fails (too little data)
        forecasts["rf"] = lr_df.copy()

    # Combined (simple average)
    comb_df = pd.DataFrame({
        "month": months_labels,
        "revenue": (forecasts["linear"]["revenue"].values + forecasts["rf"]["revenue"].values) / 2,
        "expenses": (forecasts["linear"]["expenses"].values + forecasts["rf"]["expenses"].values) / 2
    })
    comb_df["profit"] = comb_df["revenue"] - comb_df["expenses"]
    forecasts["combined"] = comb_df

    return forecasts

# --- Insights & alerts ---
def generate_insights(df):
    """
    Simple heuristics to produce human-readable insights & alerts:
    - expense_ratio alerts
    - profit trend (slope)
    - volatility (std dev of profit)
    """
    insights = []
    alerts = []

    if df.empty:
        return {"insights": ["No data available"], "alerts": []}

    # expense ratio per month
    df = df.copy()
    df["expense_ratio"] = df["expenses"] / df["revenue"].replace(0, np.nan)
    high_expense = df[df["expense_ratio"] > 0.8]
    if not high_expense.empty:
        months = ", ".join(high_expense["month"].astype(str).tolist())
        alerts.append(f"High expense ratio (>80%) in: {months} — review costs.")

    # profit trend via linear regression slope
    if len(df) >= 2:
        X = np.arange(len(df)).reshape(-1,1)
        lr = LinearRegression().fit(X, df["profit"])
        slope = lr.coef_[0]
        if slope > 0:
            insights.append(f"Profit is trending up (avg slope={slope:.2f} per month).")
        elif slope < 0:
            insights.append(f"Profit is trending down (avg slope={slope:.2f} per month). Consider actions to boost revenue or cut costs.")
        else:
            insights.append("Profit trend is flat.")

    # volatility
    profit_std = df["profit"].std()
    if pd.notna(profit_std):
        if profit_std > max(1.0, 0.2 * max(df["profit"].abs().median(), 1)):
            insights.append(f"Profit has high volatility (std={profit_std:.2f}). Consider smoothing costs or diversifying revenue.")
        else:
            insights.append("Profit volatility is low.")

    # best/worst month
    best = df.loc[df["profit"].idxmax()]["month"]
    worst = df.loc[df["profit"].idxmin()]["month"]
    insights.append(f"Best month: {best}. Worst month: {worst}.")

    return {"insights": insights, "alerts": alerts}

# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/finance", methods=["GET"])
def get_finance():
    records = FinanceRecord.query.all()
    return jsonify([r.to_dict() for r in records])

@app.route("/finance/add", methods=["POST"])
def add_finance():
    data = request.json
    month = data.get("month")
    revenue = data.get("revenue")
    expenses = data.get("expenses")
    if not month or revenue is None or expenses is None:
        return jsonify({"error":"month, revenue, and expenses are required"}), 400
    profit = revenue - expenses
    new_record = FinanceRecord(month=month, revenue=revenue, expenses=expenses, profit=profit)
    db.session.add(new_record)
    db.session.commit()
    return jsonify({"message":"Finance record added successfully", "record": new_record.to_dict()})

@app.route("/finance/upload", methods=["POST"])
def upload_finance():
    if "file" not in request.files:
        return jsonify({"error":"No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error":"Only CSV files allowed"}), 400
    try:
        df = pd.read_csv(file)
        required = {"month","revenue","expenses"}
        if not required.issubset(set(df.columns)):
            return jsonify({"error":"CSV must contain columns: month, revenue, expenses"}), 400
        # Optional: normalize column names
        df = df[list(required)].copy()
        df["profit"] = df["revenue"] - df["expenses"]

        # Delete old records
        FinanceRecord.query.delete()
        db.session.commit()

        # Insert new ones
        for _, row in df.iterrows():
            rec = FinanceRecord(month=str(row["month"]), revenue=float(row["revenue"]), expenses=float(row["expenses"]), profit=float(row["profit"]))
            db.session.add(rec)
        db.session.commit()

        return jsonify({"message":"CSV uploaded and replaced data successfully", "rows": len(df)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/finance/export", methods=["GET"])
def export_finance():
    # export current finance data as CSV
    records = FinanceRecord.query.all()
    df = pd.DataFrame([r.to_dict() for r in records])
    if df.empty:
        return jsonify({"error":"No data to export"}), 400
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return send_file(io.BytesIO(csv_bytes), mimetype="text/csv", as_attachment=True, attachment_filename="finance_export.csv")

@app.route("/forecast", methods=["GET"])
def get_forecast():
    records = FinanceRecord.query.all()
    df = build_df_from_records(records)
    forecasts = forecast_models(df, months=6)
    # send back each model's forecast as list of dicts
    return jsonify({
        "linear": forecasts["linear"].to_dict(orient="records"),
        "rf": forecasts["rf"].to_dict(orient="records"),
        "combined": forecasts["combined"].to_dict(orient="records")
    })

@app.route("/scenario", methods=["POST"])
def run_scenario():
    data = request.json
    rev_growth = float(data.get("revenue_growth", 0))
    exp_growth = float(data.get("expense_growth", 0))
    records = FinanceRecord.query.all()
    df = build_df_from_records(records)
    if df.empty:
        return jsonify({"scenario": [], "summary": "No data available."})
    df2 = df.copy()
    df2["revenue"] = df2["revenue"] * (1 + rev_growth)
    df2["expenses"] = df2["expenses"] * (1 + exp_growth)
    df2["profit"] = df2["revenue"] - df2["expenses"]

    avg_profit = df2["profit"].mean()
    summary = (
        f"Scenario Analysis:\n"
        f"- Revenue growth applied: {rev_growth*100:.1f}%\n"
        f"- Expense growth applied: {exp_growth*100:.1f}%\n"
        f"- Average monthly profit: ${avg_profit:,.2f}\n"
        f"Overall impact: {'positive' if avg_profit > 0 else 'negative'}."
    )

    # generate insights for the adjusted df
    insights = generate_insights(df2)

    return jsonify({
        "scenario": df2.to_dict(orient="records"),
        "summary": summary,
        "insights": insights
    })

@app.route("/insights", methods=["GET"])
def get_insights():
    records = FinanceRecord.query.all()
    df = build_df_from_records(records)
    return jsonify(generate_insights(df))

@app.route("/finance/delete_all", methods=["DELETE"])
def delete_all_records():
    FinanceRecord.query.delete()
    db.session.commit()
    return jsonify({"message":"All records deleted successfully"})

# --- Run app ---
if __name__ == "__main__":
    app.run(debug=True, port=5000)
