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
    df = pd.read_csv("data.csv")

    total_accidents = len(df)
    total_injuries = df['Injuries'].sum()

    data = df.to_dict(orient="records")

    return render_template("dashboard.html",
                           total_accidents=total_accidents,
                           total_injuries=total_injuries,
                           data=data)
# Run app
if __name__ == "__main__":
    app.run(debug=True)