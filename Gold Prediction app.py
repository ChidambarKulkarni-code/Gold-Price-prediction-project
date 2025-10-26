import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gold Price Forecasting", page_icon="💰", layout="wide")

st.title("💰 Gold Price Forecasting (INR per 10g)")
st.write("Upload your gold price CSV (with a **date** column and a **price** column). The app will create lag features, train a model, and forecast future values.")

# --- Sidebar inputs ---
with st.sidebar:
    st.header("1) Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    date_col = st.text_input("Date column name", value="date")
    price_col = st.text_input("Price column name", value="price")
    st.caption("Tip: If your file has columns like Date and Gold_Price_INR_per_10g, set those names above.")

    st.header("2) Features & Model")
    max_lag = st.slider("Max lag days", 3, 60, 30)
    train_ratio = st.slider("Train split ratio", 0.5, 0.95, 0.8, step=0.05)
    n_estimators = st.slider("RandomForest n_estimators", 100, 1000, 400, step=50)
    random_state = st.number_input("Random state", value=42, step=1)

    st.header("3) Forecast")
    horizon = st.slider("Forecast horizon (days)", 1, 60, 14)

def load_data(file, date_col, price_col):
    df = pd.read_csv(file)
    # Try a few common fallbacks for column names
    cols = {c.lower().strip(): c for c in df.columns}
    dcol = cols.get(date_col.lower().strip(), None)
    pcol = cols.get(price_col.lower().strip(), None)
    if dcol is None:
        for cand in ["date", "Date", "DATE", "timestamp"]:
            if cand in df.columns:
                dcol = cand
                break
    if pcol is None:
        for cand in ["price", "gold_price", "Gold_Price_INR_per_10g", "Close", "close"]:
            if cand in df.columns:
                pcol = cand
                break
    if dcol is None or pcol is None:
        raise ValueError("Could not infer date/price columns. Please set correct names in the sidebar.")
    out = df[[dcol, pcol]].copy()
    out.columns = ["date", "price"]
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").dropna()
    out = out[~out["price"].isna()]
    return out

def make_lag_features(df, max_lag):
    work = df.copy()
    for lag in range(1, max_lag + 1):
        work[f"lag_{lag}"] = work["price"].shift(lag)
    work = work.dropna().reset_index(drop=True)
    feature_cols = [c for c in work.columns if c.startswith("lag_")]
    return work, feature_cols

def train_model(df_lagged, feature_cols, train_ratio, n_estimators, random_state):
    n = len(df_lagged)
    split = int(n * train_ratio)
    X = df_lagged[feature_cols].values
    y = df_lagged["price"].values
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

preds = model.predict(X_test) if len(X_test) > 0 else np.array([])
metrics = {}
if len(X_test) > 0:
    # Compute RMSE without relying on the 'squared' kwarg (older sklearn compat)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, preds)
    metrics = {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}
    return model, metrics, split

def recursive_forecast(last_values, model, horizon):
    """
    last_values: list/array of latest lag values in order [lag_1, lag_2, ..., lag_k]
    We'll predict next value, then roll the window and continue.
    """
    last_values = list(last_values)
    k = len(last_values)
    forecasts = []
    for _ in range(horizon):
        x = np.array(last_values).reshape(1, -1)
        yhat = float(model.predict(x)[0])
        forecasts.append(yhat)
        # shift right and insert new yhat at position 0
        last_values = [yhat] + last_values[:-1]
    return forecasts

def plot_series(df, title):
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["price"])
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

def plot_actual_vs_pred(dates, actual, pred, title):
    fig, ax = plt.subplots()
    ax.plot(dates, actual, label="Actual")
    ax.plot(dates, pred, label="Predicted")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

# --- Main app ---
if uploaded is None:
    st.info("Upload a CSV to begin. A sample will be generated if you don't have one.")
    if st.button("Use sample synthetic data"):
        rng = pd.date_range("2020-01-01", periods=800, freq="D")
        # synthetic trend + noise
        price = 40000 + np.linspace(0, 5000, len(rng)) + np.random.normal(0, 500, len(rng))
        sample = pd.DataFrame({"date": rng, "price": price})
        st.session_state["sample_df"] = sample
        st.success("Sample data loaded (synthetic). Go to 'Model' section below.")
else:
    try:
        data = load_data(uploaded, date_col, price_col)
        st.subheader("Data Preview")
        st.write(data.head(10))
        st.caption(f"Rows: {len(data)} | Date range: {data['date'].min().date()} → {data['date'].max().date()}")

        st.subheader("Time Series Plot")
        plot_series(data, "Gold Price Over Time")

        # Build lag features
        df_lagged, feature_cols = make_lag_features(data, max_lag=max_lag)

        # Train
        model, metrics, split = train_model(df_lagged, feature_cols, train_ratio, n_estimators, random_state)

        # Show metrics
        st.subheader("Evaluation (Hold-out test)")
        if metrics:
            st.write(pd.DataFrame([metrics]))
            # Plot actual vs predicted on test
            test_dates = data["date"].iloc[len(data) - len(df_lagged) + split : len(data)].reset_index(drop=True)
            actual = df_lagged["price"].iloc[split:].reset_index(drop=True)
            pred = model.predict(df_lagged[feature_cols].iloc[split:])
            plot_actual_vs_pred(test_dates, actual, pred, "Actual vs Predicted (Test Set)")
        else:
            st.warning("Not enough data for a test split at current settings. Increase data or adjust train ratio.")

        # Forecast
        st.subheader("Forecast")
        last_row = df_lagged.iloc[-1]
        last_lags = [last_row[f"lag_{i}"] for i in range(1, max_lag + 1)]
        fc_values = recursive_forecast(last_lags, model, horizon=horizon)

        last_date = data["date"].iloc[-1]
        fc_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]
        df_fc = pd.DataFrame({"date": fc_dates, "forecast": fc_values})

        st.write(df_fc)

        # Plot forecast appended to history
        hist_tail = data.tail(100).copy()
        fig, ax = plt.subplots()
        ax.plot(hist_tail["date"], hist_tail["price"], label="History (last 100)")
        ax.plot(df_fc["date"], df_fc["forecast"], label="Forecast")
        ax.set_title("History + Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        st.download_button(
            "Download forecast as CSV",
            data=df_fc.to_csv(index=False).encode("utf-8"),
            file_name="gold_price_forecast.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

st.caption("Note: This app uses a simple lag-feature RandomForest approach for speed and easy deployment on Streamlit Cloud.")
