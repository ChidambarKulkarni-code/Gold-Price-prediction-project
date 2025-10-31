

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Gold Price Predictor", layout="wide")
st.title("🪙 Gold Price Viewer")


# Fixed path
CSV_PATH = "gold_price_predictions_final.csv" 

# ------------------- Load Data -------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Detect date column
    date_cols = [c for c in df.columns if c.lower() in ["date", "day", "timestamp"]]
    if not date_cols:
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date"}, inplace=True)
        date_col = "Date"
    else:
        date_col = date_cols[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: "Date"})
    return df

# ------------------- Main -------------------
try:
    df = load_data(CSV_PATH)
except Exception as e:
    st.error(f"❌ Could not load `{CSV_PATH}`. Make sure it exists next to this app.")
    with st.expander("Error details"):
        st.exception(e)
    st.stop()

st.success(f"Loaded {len(df):,} rows from the CSV.")
with st.expander("Peek at file (last 10 rows)"):
    st.dataframe(df.tail(10), use_container_width=True)

# Detect available dates
available_dates = sorted(df["Date"].dt.date.unique())
if not available_dates:
    st.error("No valid dates found in the file.")
    st.stop()

min_date, max_date = available_dates[0], available_dates[-1]

# Date picker (only necessary control)
selected_date = st.date_input(
    "Select a date to view data",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

# Filter and show
filtered = df[df["Date"].dt.date == selected_date]
if filtered.empty:
    st.warning(f"No data available for {selected_date}. Try another date.")
else:
    st.subheader(f"Data for {selected_date}")
    st.dataframe(filtered, use_container_width=True)



