# Gold Price Forecasting – Streamlit App

A plug-and-play Streamlit app to explore, model, and forecast Gold Price (INR per 10g) time series using simple lag features + RandomForest.

## File structure

```
gold_price_streamlit_app/
├─ app.py
├─ requirements.txt
└─ README.md
```

## Expected CSV format

- A **date** column (parseable by pandas)
- A **price** column (numeric, e.g., INR per 10g)
- You can rename these in the sidebar if your columns are different (e.g., `Gold_Price_INR_per_10g`).

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push these files to a **public GitHub repo** (e.g., `gold-price-forecasting-app`).
2. Go to Streamlit Cloud → **Deploy app** → Connect to your repo.
3. Set:
   - **Main file path**: `app.py`
   - **Python version**: 3.10 or 3.11
4. Click **Deploy**.

## Notes

- The model is a fast baseline. For production, consider advanced models (Prophet, SARIMA, XGBoost) and richer features (seasonality, macro variables, FX, CPI, holidays).