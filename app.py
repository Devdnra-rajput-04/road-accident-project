import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

# ── Load CSV once ─────────────────────────────────────────────────────────────
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.month
        return df
    except FileNotFoundError:
        return pd.DataFrame()

DF = load_data()

# ── Load ML model ─────────────────────────────────────────────────────────────
MODEL = None
try:
    with open(MODEL_PATH, "rb") as f:
        MODEL = pickle.load(f)
except Exception as e:
    print(f"Warning: could not load model — {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def apply_filters(df, year=None, weather=None, states=None):
    """Filter the DataFrame based on user selections."""
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
    fatal   = int(sev.get("Fatal",   0))
    serious = int(sev.get("Serious", 0))
    minor   = int(sev.get("Minor",   0))

    bike_kws = ["Motorcycle", "Bicycle"]
    bike_injuries = int(
        df[df["Primary_Vehicle"].isin(bike_kws)]["Injuries"].sum()
        if "Primary_Vehicle" in df.columns else 0
    )
    return dict(
        total_accidents=total,
        total_injuries=total_injuries,
        fatal=fatal,
        serious=serious,
        minor=minor,
        bike_injuries=bike_injuries,
    )


def build_charts(df):
    # Monthly trend
    monthly_trend = {}
    if "Month" in df.columns:
        monthly_trend = df.groupby("Month").size().to_dict()

    # Road type
    road_types = {}
    if "Road_Type" in df.columns:
        road_types = df["Road_Type"].value_counts().head(6).to_dict()

    # Road surface / Weather (we map weather here as surface equivalent)
    road_surface = {}
    if "Weather" in df.columns:
        road_surface = df["Weather"].value_counts().to_dict()

    # Gender
    gender = {}
    if "Driver_Gender" in df.columns:
        gender = df["Driver_Gender"].value_counts().to_dict()

    # Light conditions
    light = {}
    if "Light_Condition" in df.columns:
        light = df["Light_Condition"].value_counts().to_dict()

    # Vehicle type
    vehicle_counts = {}
    if "Primary_Vehicle" in df.columns:
        vehicle_counts = df["Primary_Vehicle"].value_counts().to_dict()

    # Cause breakdown
    cause_counts = {}
    if "Cause" in df.columns:
        cause_counts = df["Cause"].value_counts().head(5).to_dict()

    # Yearly trend (for multi-year view)
    yearly_trend = {}
    if "Year" in df.columns:
        yearly_trend = df.groupby("Year").size().to_dict()

    return dict(
        monthly_trend=monthly_trend,
        road_types=road_types,
        road_surface=road_surface,
        gender=gender,
        light=light,
        vehicle_counts=vehicle_counts,
        cause_counts=cause_counts,
        yearly_trend=yearly_trend,
    )


# ── Routes ────────────────────────────────────────────────────────────────────
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
    tables = top200.to_html(
        classes="data-table",
        index=False,
        border=0,
        table_id="main-table"
    )
    return render_template("data.html", tables=tables)

@app.route("/api/data")
def api_data():
    try:
        df_copy = DF.copy()

        # datetime fix
        for col in df_copy.columns:
            if df_copy[col].dtype == 'datetime64[ns]':
                df_copy[col] = df_copy[col].astype(str)

        # NaN fix
        df_copy = df_copy.fillna("")

        return jsonify(df_copy.to_dict(orient="records"))

    except Exception as e:
        return {"error": str(e)}

# ── API: Real Filter Endpoint ─────────────────────────────────────────────────
@app.route("/api/filter")
def api_filter():
    year    = request.args.get("year")       # e.g. "2021" or ""
    weather = request.args.get("weather", "All")
    states  = request.args.getlist("state")  # multi-value: ?state=Bihar&state=Delhi
    if not states:
        states = ["All"]

    filtered = apply_filters(DF.copy(), year=year or None, weather=weather, states=states)
    stats  = build_stats(filtered)
    charts = build_charts(filtered)
    return jsonify({**stats, **charts})


# ── API: ML Prediction ────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    clf       = MODEL["clf"]
    encoders  = MODEL["encoders"]
    le_target = MODEL["le_target"]
    features  = MODEL["features"]

    try:
        row = {}
        cat_cols = ["Road_Type", "Weather", "Light_Condition", "Driver_Gender",
                    "Helmet_Seatbelt_Used", "Primary_Vehicle", "Cause"]
        num_cols = ["Driver_Age", "Estimated_Speed_kmh", "Vehicles_Involved", "Year"]

        for col in cat_cols:
            val = data.get(col, "")
            le  = encoders[col]
            if val in le.classes_:
                row[col] = le.transform([val])[0]
            else:
                row[col] = 0

        for col in num_cols:
            row[col] = float(data.get(col, 0))

        X = np.array([[row[f] for f in features]])
        pred_idx  = clf.predict(X)[0]
        pred_prob = clf.predict_proba(X)[0]
        severity  = le_target.inverse_transform([pred_idx])[0]

        probabilities = {
            cls: round(float(prob), 3)
            for cls, prob in zip(le_target.classes_, pred_prob)
        }

        return jsonify({"severity": severity, "probabilities": probabilities})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_ENV") == "development")
