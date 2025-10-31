
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta, datetime

st.set_page_config(page_title="Gold Price Predictor (Fixed CSV)", layout="wide")
st.title("🪙 Gold Price Predictor (Fixed CSV only)")
st.caption("Data source is locked to **/mnt/data/gold_price_predictions_final.csv** — no other files allowed.")

CSV_PATH = "/mnt/data/gold_price_predictions_final.csv"

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Try to find date column
    date_cols = [c for c in df.columns if c.lower() in ["date", "day", "timestamp"]]
    if len(date_cols) == 0:
        # try infer datetime from index if no explicit column
        df = df.copy()
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date"}, inplace=True)
        date_col = "Date"
    else:
        date_col = date_cols[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: "Date"})

    # Identify price column with best guess
    candidates = ["Actual_Price_INR", "Gold_Price", "Gold Price", "Price", "Close", "Close_Price", "Close Price"]
    price_col = None
    for c in candidates:
        if c in df.columns:
            price_col = c
            break

    if price_col is None:
        # if only one numeric column, assume it
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 1:
            price_col = numeric_cols[0]
        else:
            raise ValueError(
                "Could not find a price column. Expected one of: "
                + ", ".join(candidates)
            )

    df = df[["Date", price_col]].rename(columns={price_col: "Price"})
    # Clean
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"]).reset_index(drop=True)

    # Ensure daily frequency (fill gaps if any)
    df = df.set_index("Date").asfreq("D")
    df["Price"] = df["Price"].interpolate(method="time").bfill().ffill()
    df = df.reset_index()

    return df

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lag_1"] = df["Price"].shift(1)
    df["lag_7"] = df["Price"].shift(7)
    df["lag_14"] = df["Price"].shift(14)
    df["lag_30"] = df["Price"].shift(30)
    df["roll_7"] = df["Price"].rolling(7).mean()
    df["roll_14"] = df["Price"].rolling(14).mean()
    df["roll_30"] = df["Price"].rolling(30).mean()

    # Calendar features
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day

    df = df.dropna().reset_index(drop=True)
    return df

def train_model(df_feat: pd.DataFrame):
    # Train/test split by time (last 20% as test)
    split_idx = int(len(df_feat) * 0.8)
    train = df_feat.iloc[:split_idx]
    test  = df_feat.iloc[split_idx:]

    X_train = train.drop(columns=["Date", "Price"])
    y_train = train["Price"]
    X_test  = test.drop(columns=["Date", "Price"])
    y_test  = test["Price"]

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
    return model, metrics, train, test

def predict_at_date(model, history_df: pd.DataFrame, target_date: pd.Timestamp) -> float:
    """
    Predict price for a given date. If the date is in-sample, we use its features.
    If it's future, we recursively forecast day by day using last known/forecasted values.
    """
    df = history_df.copy().set_index("Date")
    last_date = df.index.max()

    if target_date <= last_date:
        # In-sample prediction
        row = df.loc[[target_date]].copy()
        features = row.drop(columns=["Price"]).values
        return float(model.predict(features)[0])

    # Future recursive forecasting
    cur_date = last_date + timedelta(days=1)
    df_future = df.copy()

    while cur_date <= target_date:
        # Build a temporary row using lagged values from df_future
        tmp = {"Date": cur_date}

        # Create lags/rollings: we need a rolling window from df_future
        # For simplicity, compute on the fly using last available values
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

        last_feat = hist[hist["Date"] == (cur_date - timedelta(days=1))].drop(columns=["Price"])
        if last_feat.empty:
            # If we cannot compute features (too early), break
            break

        # For the current day, copy yesterday's features and update calendar
        x = last_feat.copy()
        x.loc[:, "dayofweek"] = cur_date.weekday()
        x.loc[:, "month"] = cur_date.month
        x.loc[:, "day"] = cur_date.day

        # Model prediction
        yhat = float(model.predict(x.values)[0])

        # Append to df_future
        df_future.loc[cur_date, "Price"] = yhat

        cur_date += timedelta(days=1)

    return float(df_future.loc[target_date, "Price"])

# Load and prep
try:
    raw_df = load_data(CSV_PATH)
except Exception as e:
    st.error(f"Failed to load the fixed CSV: {e}")
    st.stop()

st.success(f"Loaded {len(raw_df):,} rows from the fixed CSV.")
with st.expander("Peek at data (tail)"):
    st.dataframe(raw_df.tail(20), use_container_width=True)

feat_df = make_features(raw_df)

# Train model
model, metrics, train, test = train_model(feat_df)

# Show metrics
mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("MAE", f"{metrics['MAE']:.2f}")
mcol2.metric("RMSE", f"{metrics['RMSE']:.2f}")
mcol3.metric("R²", f"{metrics['R2']:.3f}")

st.markdown("---")
st.subheader("🔮 Forecast")
min_date = raw_df["Date"].min().date()
max_date = raw_df["Date"].max().date()
default_future = max_date + timedelta(days=7)

# Only allow choosing a date; NO upload widgets anywhere
target = st.date_input(
    "Pick a date to predict (in-sample or future)",
    value=default_future,
    min_value=min_date,
    max_value=max_date + timedelta(days=365)  # allow up to +1 year
)

if st.button("Predict for selected date"):
    tstamp = pd.to_datetime(target)
    try:
        yhat = predict_at_date(model, feat_df, tstamp)
        st.success(f"Predicted Price on {tstamp.date().isoformat()}: **{yhat:,.2f}**")
    except Exception as e:
        st.error(f"Could not predict for {tstamp.date()}: {e}")

st.caption("Note: This app uses a simple RandomForest with lag/rolling features for quick, no-upload predictions. For production, consider a dedicated time-series model and hyperparameter tuning.")
