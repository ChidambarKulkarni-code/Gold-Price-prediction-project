import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

# -------------------- Streamlit Page Setup --------------------
st.set_page_config(page_title="Gold Price Forecasting (Fixed Dataset)", layout="wide")
st.title("🏆 Gold Price Forecasting (Using Fixed Dataset + Date Input)")

st.markdown("""
This app predicts **Indian Gold Prices (INR)** using a preloaded dataset.  
You can select any date — past or future — and the model will forecast the expected gold price trend.
""")

# -------------------- Load Dataset --------------------
DATA_PATH = "gold_price_predictions_final.csv"   # Must be in the same repo/folder

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    
    # Detect date column
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if not date_cols:
        raise ValueError("❌ No date column found. Expected one column named 'Date'.")
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Detect gold price column (target)
    target_cols = [c for c in df.columns if "gold" in c.lower()]
    if not target_cols:
        raise ValueError("❌ No gold price column found. Expected one column containing 'Gold'.")
    target_col = target_cols[0]

    df = df[[date_col, target_col]].dropna().copy()
    df = df.sort_values(by=date_col)
    df.rename(columns={date_col: "Date", target_col: "GoldPrice"}, inplace=True)

    # Time-related features
    df["t"] = (df["Date"] - df["Date"].min()).dt.days
    df["month"] = df["Date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df

try:
    df = load_data(DATA_PATH)
    st.success(f"✅ Loaded dataset: {DATA_PATH}")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

st.caption(f"Rows: {len(df)} | Date Range: {df['Date'].min().date()} → {df['Date'].max().date()}")

# -------------------- Data Preview --------------------
with st.expander("📊 Data Preview and EDA", expanded=True):
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Gold Price Over Time")
        fig, ax = plt.subplots()
        ax.plot(df["Date"], df["GoldPrice"], color="gold")
        ax.set_xlabel("Date"); ax.set_ylabel("Gold Price (INR)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col2:
        st.write("### Distribution of Gold Price")
        fig, ax = plt.subplots()
        ax.hist(df["GoldPrice"], bins=25, color="orange", alpha=0.8)
        ax.set_xlabel("Gold Price (INR)")
        st.pyplot(fig)

# -------------------- Train/Test Split --------------------
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df[["t", "month_sin", "month_cos"]]
y_train = train_df["GoldPrice"]
X_test = test_df[["t", "month_sin", "month_cos"]]
y_test = test_df["GoldPrice"]

# -------------------- Model Building --------------------
degree = st.sidebar.slider("Polynomial Degree (Time Trend)", 1, 5, 2)
alpha = st.sidebar.slider("Ridge Regularization (Alpha)", 0.0, 10.0, 1.0, 0.1)

poly = PolynomialFeatures(degree=degree, include_bias=False)
pre = ColumnTransformer([
    ("poly_t", poly, ["t"]),
    ("season", "passthrough", ["month_sin", "month_cos"])
])

model = Pipeline([
    ("pre", pre),
    ("ridge", Ridge(alpha=alpha, random_state=42))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("MSE (↓)", f"{mse:,.2f}")
with col2:
    st.metric("R² Score (↑)", f"{r2:.4f}")

fig, ax = plt.subplots(figsize=(8,3))
ax.plot(test_df["Date"], y_test, label="Actual", color="black")
ax.plot(test_df["Date"], y_pred, label="Predicted", color="purple")
ax.set_title("Actual vs Predicted Gold Prices (Test Period)")
ax.legend(); ax.grid(True, alpha=0.3)
st.pyplot(fig)

# -------------------- Prediction for User Date --------------------
st.subheader("🗓️ Predict Gold Price for a Given Date")

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

st.caption(f"Training Period: {min_date} → {max_date}")
user_date = st.date_input("Select a date to predict", value=max_date)

# Create date-based features
user_date_dt = pd.to_datetime(user_date)
t_val = (user_date_dt - df["Date"].min()).days
month = user_date_dt.month
input_features = pd.DataFrame({
    "t": [t_val],
    "month_sin": [np.sin(2 * np.pi * month / 12)],
    "month_cos": [np.cos(2 * np.pi * month / 12)]
})

predicted_price = float(model.predict(input_features)[0])
st.success(f"💰 Predicted Gold Price for {user_date}: ₹{predicted_price:,.2f}")

with st.expander("ℹ️ About the Prediction"):
    st.markdown("""

