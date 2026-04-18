import os
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ===== Load Data =====
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data.csv")

def load_data():
    try:
        df = pd.read_csv(DATA_PATH)

        # Date handling
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year

        return df
    except:
        return pd.DataFrame()

DF = load_data()

# ===== STATS (MATCH WITH EXCEL SUM) =====
def build_stats(df):

    total_records = len(df)
    total_injuries = int(df["Injuries"].sum())

    fatal   = int(df[df["Accident_Severity"] == "Fatal"]["Injuries"].sum())
    serious = int(df[df["Accident_Severity"] == "Serious"]["Injuries"].sum())
    minor   = int(df[df["Accident_Severity"] == "Minor"]["Injuries"].sum())

    bike_injuries = int(
        df[df["Primary_Vehicle"].isin(["Motorcycle", "Bicycle"])]["Injuries"].sum()
    )

    return {
        "total_records": total_records,
        "total_injuries": total_injuries,
        "fatal": fatal,
        "serious": serious,
        "minor": minor,
        "bike_injuries": bike_injuries,
    }

# ===== CHARTS (ALL SUM BASED) =====
def build_charts(df):

    monthly_trend = df.groupby("Month")["Injuries"].sum()
    road_types = df.groupby("Road_Type")["Injuries"].sum()
    road_surface = df.groupby("Weather")["Injuries"].sum()
    gender = df.groupby("Driver_Gender")["Injuries"].sum()
    light = df.groupby("Light_Condition")["Injuries"].sum()
    vehicle = df.groupby("Primary_Vehicle")["Injuries"].sum()
    cause = df.groupby("Cause")["Injuries"].sum()
    yearly = df.groupby("Year")["Injuries"].sum()

    def safe(d):
        return {str(k): int(v) for k, v in d.to_dict().items()}

    return {
        "monthly_trend": safe(monthly_trend),
        "road_types": safe(road_types),
        "road_surface": safe(road_surface),
        "gender": safe(gender),
        "light": safe(light),
        "vehicle_counts": safe(vehicle),
        "cause_counts": safe(cause),
        "yearly_trend": safe(yearly),
    }

# ===== FILTER API =====
@app.route("/api/filter")
def filter_data():
    df = DF.copy()

    year = request.args.get("year")
    weather = request.args.get("weather")
    state = request.args.get("state")

    if year and year != "All":
        df = df[df["Year"] == int(year)]

    if weather and weather != "All":
        df = df[df["Weather"] == weather]

    if state and state != "All":
        df = df[df["State"] == state]

    stats = build_stats(df)
    charts = build_charts(df)

    return jsonify({**stats, **charts})

# ===== ROUTES =====
@app.route("/")
def home():
    stats = build_stats(DF)
    return render_template("index.html", **stats)

@app.route("/dashboard")
def dashboard():
    stats = build_stats(DF)
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

# ===== RUN =====
if __name__ == "__main__":
    app.run(debug=True)