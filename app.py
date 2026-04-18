import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Paths ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

# ── Load CSV ──────────────────────────────────────────
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.month
        return df
    except:
        return pd.DataFrame()

DF = load_data()

# ── Load Model ────────────────────────────────────────
MODEL = None
try:
    with open(MODEL_PATH, "rb") as f:
        MODEL = pickle.load(f)
except:
    print("Model not loaded")

# ── Helpers ───────────────────────────────────────────
def apply_filters(df, year=None, weather=None, states=None):
    if year:
        df = df[df["Year"] == int(year)]

    if weather and weather != "All":
        df = df[df["Weather"] == weather]

    if states and "All" not in states:
        df = df[df["State"].isin(states)]

    return df


def build_stats(df):
    total = len(df)

    total_injuries = int(df["Injuries"].sum()) if "Injuries" in df.columns else total

    sev = df["Accident_Severity"].value_counts() if "Accident_Severity" in df.columns else {}

    fatal   = int(sev.get("Fatal", 0))
    serious = int(sev.get("Serious", 0))
    minor   = int(sev.get("Minor", 0))

    bike_kws = ["Motorcycle", "Bicycle"]

    bike_injuries = int(
        df[df["Primary_Vehicle"].isin(bike_kws)]["Injuries"].sum()
        if "Primary_Vehicle" in df.columns else 0
    )

    return {
        "total_injuries": int(total_injuries),
        "fatal": int(fatal),
        "serious": int(serious),
        "minor": int(minor),
        "bike_injuries": int(bike_injuries),
    }


# 🔥 JSON SAFE FUNCTION (VERY IMPORTANT)
def safe_dict(d):
    return {str(k): int(v) for k, v in d.items()}


def build_charts(df):
    monthly_trend = safe_dict(df.groupby("Month").size().to_dict()) if "Month" in df.columns else {}
    road_types    = safe_dict(df["Road_Type"].value_counts().head(6).to_dict()) if "Road_Type" in df.columns else {}
    road_surface  = safe_dict(df["Weather"].value_counts().to_dict()) if "Weather" in df.columns else {}
    gender        = safe_dict(df["Driver_Gender"].value_counts().to_dict()) if "Driver_Gender" in df.columns else {}
    light         = safe_dict(df["Light_Condition"].value_counts().to_dict()) if "Light_Condition" in df.columns else {}
    vehicle       = safe_dict(df["Primary_Vehicle"].value_counts().to_dict()) if "Primary_Vehicle" in df.columns else {}
    cause         = safe_dict(df["Cause"].value_counts().head(5).to_dict()) if "Cause" in df.columns else {}
    yearly        = safe_dict(df.groupby("Year").size().to_dict()) if "Year" in df.columns else {}

    return {
        "monthly_trend": monthly_trend,
        "road_types": road_types,
        "road_surface": road_surface,
        "gender": gender,
        "light": light,
        "vehicle_counts": vehicle,
        "cause_counts": cause,
        "yearly_trend": yearly,
    }


# ── Routes ────────────────────────────────────────────
@app.route("/")
def home():
    stats = build_stats(DF)
    return render_template("index.html", **stats)


@app.route("/dashboard")
def dashboard():
    stats  = build_stats(DF)
    charts = build_charts(DF)
    return render_template("dashboard.html", **stats, **charts)


@app.route("/data")
def data():
    top200 = DF.head(200)
    tables = top200.to_html(classes="data-table", index=False)
    return render_template("data.html", tables=tables)


# 🔥 FIXED API DATA (LIMIT ADDED)
@app.route("/api/data")
def api_data():
    try:
        df_copy = DF.copy().head(500)   # 🚨 IMPORTANT LIMIT

        df_copy = df_copy.fillna("")

        for col in df_copy.columns:
            if str(df_copy[col].dtype).startswith("datetime"):
                df_copy[col] = df_copy[col].astype(str)

        return jsonify(df_copy.to_dict(orient="records"))

    except Exception as e:
        return {"error": str(e)}


# 🔥 MAIN FILTER API (DASHBOARD USE)
@app.route("/api/filter")
def api_filter():
    year    = request.args.get("year")
    weather = request.args.get("weather", "All")
    states  = request.args.getlist("state")

    if not states:
        states = ["All"]

    df = apply_filters(DF.copy(), year, weather, states)

    stats  = build_stats(df)
    charts = build_charts(df)

    return jsonify({**stats, **charts})


# ── ML Prediction ─────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()

    try:
        clf       = MODEL["clf"]
        encoders  = MODEL["encoders"]
        le_target = MODEL["le_target"]
        features  = MODEL["features"]

        row = {}

        cat_cols = ["Road_Type", "Weather", "Light_Condition", "Driver_Gender",
                    "Helmet_Seatbelt_Used", "Primary_Vehicle", "Cause"]

        num_cols = ["Driver_Age", "Estimated_Speed_kmh", "Vehicles_Involved", "Year"]

        for col in cat_cols:
            val = data.get(col, "")
            le  = encoders[col]
            row[col] = le.transform([val])[0] if val in le.classes_ else 0

        for col in num_cols:
            row[col] = float(data.get(col, 0))

        X = np.array([[row[f] for f in features]])

        pred_idx  = clf.predict(X)[0]
        pred_prob = clf.predict_proba(X)[0]

        severity = le_target.inverse_transform([pred_idx])[0]

        probabilities = {
            str(cls): float(round(prob, 3))
            for cls, prob in zip(le_target.classes_, pred_prob)
        }

        return jsonify({"severity": severity, "probabilities": probabilities})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── Run ───────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)