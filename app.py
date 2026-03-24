from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

# 🔹 Load dataset
df = pd.read_csv("data.csv")

# 🔹 HOME PAGE
@app.route("/")
def home():
    total_accidents = len(df)
    total_injuries = df["Injuries"].sum()

    return render_template("index.html",
                           total_accidents=total_accidents,
                           total_injuries=total_injuries)

# 🔹 DATA PAGE
@app.route("/data")
def data():
    # sirf first 100 rows dikhayenge (fast load)
    data_html = df.head(500).to_html(classes='table', index=False)

    return f"""
    <h2 style='text-align:center;'>Dataset Preview (Top 100 Rows)</h2>
    <div style='padding:20px'>
        {data_html}
    </div>
    <div style='text-align:center; margin-top:20px;'>
        <a href="/">⬅ Back to Home</a>
    </div>
    """

# 🔹 DASHBOARD PAGE
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# 🔹 RUN APP
if __name__ == "__main__":
    app.run(debug=True)