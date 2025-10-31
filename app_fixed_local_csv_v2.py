
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta

st.set_page_config(page_title="Gold Price Predictor (Fixed CSV)", layout="wide")
st.title("🪙 Gold Price Predictor (Fixed CSV only)")
st.caption("Data source is locked to **gold_price_predictions_final.csv** — no other files allowed.")

CSV_PATH = "gold_price_predictions_final.csv"  # Local file next to this script

# -----------------------------
# Utilities
# -----------------------------
def friendly_error_box(e: Exception):
    st.error(
        "Something went wrong while training or predicting. "
        "See details below and fix the data or try again."
    )
    with st.expander("Show technical details"):
        st.exception(e)

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Detect date column
    date_cols = [c for c in df.columns if c.lower() in ["date", "day", "timestamp"]]
    if len(date_cols) == 0:
        df = df.copy()
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date"}, inplace=True)
        date_col = "Date"
    else:
        date_col = date_cols[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: "Date"})

    # Detect price column
    candidates = ["Actual_Price_INR", "Gold_Price", "Gold Price", "Price", "Close", "Close_Price", "Close Price"]
    price_col = None
    for c in candidates:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 1:
            price_col = numeric_cols[0]
        else:
            raise ValueError("Could not find a price column. Expected one of: " + ", ".join(candidates))

    df = df[["Date", price_col]].rename(columns={price_col: "Price"})
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"]).reset_index(drop=True)

    # Regularize daily frequency and fill gaps
    df = df.set_index("Date").asfreq("D")
    df["Price"] = df["Price"].interpolate(method="time").bfill().ffill()
    df = df.reset_index()
    return df

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Lags
    df["lag_1"] = df["Price"].shift(1)
    df["lag_7"] = df["Price"].shift(7)
    df["lag_14"] = df["Price"].shift(14)
    df["lag_30"] = df["Price"].shift(30)
    # Rolling means
    df["roll_7"] = df["Price"].rolling(7).mean()
    df["roll_14"] = df["Price"].rolling(14).mean()
    df["roll_30"] = df["Price"].rolling(30).mean()
    # Calendars
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day

    df = df.dropna().reset_index(drop=True)
    return df

def train_model(df_feat: pd.DataFrame):
    # Basic sanity checks
    if len(df_feat) < 100:
        st.warning(f"Dataset is quite small after feature engineering ({len(df_feat)} rows). "
                   "Results may be unstable; consider providing more history.")
    split_idx = int(len(df_feat) * 0.8)
    if split_idx <= 1 or (len(df_feat) - split_idx) < 1:
        raise ValueError("Not enough rows to split into train/test after feature engineering. "
                         "Try adding more data or reducing the lag/rolling windows.")

    train = df_feat.iloc[:split_idx]
    test  = df_feat.iloc[split_idx:]

    X_train = train.drop(columns=["Date", "Price"])
    y_train = train["Price"].astype(float)
    X_test  = test.drop(columns=["Date", "Price"])
    y_test  = test["Price"].astype(float)

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    # Avoid sklearn's 'squared' kw for broad compatibility
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)

    metrics = {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}
    return model, metrics, train, test

def predict_at_date(model, history_df: pd.DataFrame, target_date: pd.Timestamp) -> float:
    df = history_df.copy().set_index("Date")
    last_date = df.index.max()

    # In-sample prediction: ensure exact match exists after dropna
    if target_date in df.index and target_date <= last_date:
        row = df.loc[[target_date]].copy()
        X = row.drop(columns=["Price"]).values
        return float(model.predict(X)[0])

    # Future recursive forecasting
    cur_date = last_date + timedelta(days=1)
    df_future = df.copy()

    while cur_date <= target_date:
        # Build features using the up-to-date df_future
        hist = df_future.copy().reset_index()
        hist["lag_1"] = hist["Price"].shift(1)
        hist["lag_7"] = hist["Price"].shift(7)
        hist["lag_14"] = hist["Price"].shift(14)
        hist["lag_30"] = hist["Price"].shift(30)
        hist["roll_7"] = hist["Price"].rolling(7).mean()
        hist["roll_14"] = hist["Price"].rolling(14).mean()
        hist["roll_30"] = hist["Price"].rolling(30).mean()
        hist["dayofweek"] = hist["Date"].dt.dayofweek
        hist["month"] = hist["Date"].dt.month
        hist["day"] = hist["Date"].dt.day

        # yesterday's row as feature base
        ref_day = cur_date - timedelta(days=1)
        last_feat = hist[hist["Date"] == ref_day].drop(columns=["Price"])
        if last_feat.empty:
            raise ValueError("Insufficient history to compute features for recursive forecast.")

        X = last_feat.copy()
        X.loc[:, "dayofweek"] = cur_date.weekday()
        X.loc[:, "month"] = cur_date.month
        X.loc[:, "day"] = cur_date.day

        yhat = float(model.predict(X.values)[0])
        df_future.loc[cur_date, "Price"] = yhat

        cur_date += timedelta(days=1)

    if target_date not in df_future.index:
        raise ValueError("Target date could not be found/generated during recursive forecast.")
    return float(df_future.loc[target_date, "Price"])

# -----------------------------
# Main
# -----------------------------
try:
    raw_df = load_data(CSV_PATH)
except Exception as e:
    st.error(f"Failed to load the fixed CSV `{CSV_PATH}`.")
    with st.expander("Show technical details"):
        st.exception(e)
    st.stop()

st.success(f"Loaded {len(raw_df):,} rows from the fixed CSV.")
with st.expander("Peek at data (tail)"):
    st.dataframe(raw_df.tail(20), use_container_width=True)

# Feature engineering
feat_df = make_features(raw_df)

# Train & Metrics
try:
    model, metrics, train, test = train_model(feat_df)
except Exception as e:
    friendly_error_box(e)
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{metrics['MAE']:.2f}")
m2.metric("RMSE", f"{metrics['RMSE']:.2f}")
m3.metric("R²", f"{metrics['R2']:.3f}")

st.markdown("---")
st.subheader("🔮 Forecast")

min_date = raw_df["Date"].min().date()
max_date = raw_df["Date"].max().date()
default_future = max_date + timedelta(days=7)

target = st.date_input(
    "Pick a date to predict",
    value=default_future,
    min_value=min_date,
    max_value=max_date + timedelta(days=365)
)

if st.button("Predict for selected date"):
    tstamp = pd.to_datetime(target)
    try:
        yhat = predict_at_date(model, feat_df, tstamp)
        st.success(f"Predicted Price on {tstamp.date().isoformat()}: **{yhat:,.2f}**")
    except Exception as e:
        friendly_error_box(e)

st.caption("Note: RMSE is computed as sqrt(MSE) for compatibility with older scikit-learn versions.")
