# 🚗 Road Accident Analysis Project

## What was fixed
- **Filter panel now uses real data** via `/api/filter` endpoint — selecting Bihar + Rain fetches actual filtered CSV rows
- **Column names corrected**: `Weather`, `Driver_Gender`, `Primary_Vehicle`, `Light_Condition` (matching your data.csv)
- **Year slider has a "Clear Year" button** — previously the slider defaulted to 2018 and silently filtered everything
- **6 new charts added**: Cause breakdown, Year-wise trend, Vehicle counts from real filtered data
- **ML prediction panel** added — Random Forest model trained on 200k rows predicts Fatal/Serious/Minor severity
- **All KPIs and charts update together** when you click "Analyze & Update Charts"

## Deploy to Render
1. Put `data.csv` in the project root (same folder as `app.py`)
2. Push to GitHub
3. On Render: New Web Service → connect repo → set Start Command: `gunicorn app:app`
4. Deploy!

## Project Structure
```
project/
├── app.py              ← Flask app with /api/filter + /api/predict
├── data.csv            ← Your 200k row dataset
├── requirements.txt    ← flask, pandas, gunicorn, scikit-learn, numpy
├── Procfile            ← web: gunicorn app:app
├── model/
│   └── model.pkl       ← Trained RandomForest model
├── templates/
│   ├── index.html
│   ├── dashboard.html  ← Fixed: real API calls
│   └── data.html
└── static/
    └── style.css
```

## Author
Devendra Rajput
