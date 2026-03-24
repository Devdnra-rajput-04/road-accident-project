from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Dashboard page
@app.route("/dashboard")
def dashboard():
    # CSV file read
    df = pd.read_csv("data.csv")

    # Total accidents = total rows
    total_accidents = len(df)

    # Total injuries (column name same hona chahiye)
    total_injuries = df['Injuries'].sum()

    return render_template("dashboard.html",
                           total_accidents=total_accidents,
                           total_injuries=total_injuries)

# Run app
if __name__ == "__main__":
    app.run(debug=True)