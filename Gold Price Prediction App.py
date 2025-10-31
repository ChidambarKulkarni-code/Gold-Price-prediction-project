from pathlib import Path

CSV_PATH = "gold_price_predictions_final.csv"  

code = r'''
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Gold Price Predictor", page_icon="🪙", layout="wide")
st.title("🪙 Gold Price Predictor")


# ---- Fixed file names (place one of these next to this script) ----
CANDIDATE_FILES = [
    "gold_price_predictions_final.csv",
    "gold_price_predictions_final.xlsx",
    "gold_price_predictions_final.xls",
]

def find_file():
    for name in CANDIDATE_FILES:
        if Path(name).exists():
            return name
    return None

FILE_PATH = find_file()
if FILE_PATH is None:
    st.error("❌ Could not find the data file. Expected one of: "
             + ", ".join(CANDIDATE_FILES))
    st.stop()

st.info(f"Using data file: **{FILE_PATH}**")

@st.cache_data(show_spinner=False)
def load_any(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return df

def detect_date_column(df: pd.DataFrame):
    # First pass: common names
    common = ["date", "day", "timestamp", "ts", "trade_date"]
    for c in df.columns:
        if c.lower() in common or c.lower() == "date":
            return c
    # Second pass: any column that parses well to datetime
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.9:  # at least 90% parse success
                return c
        except Exception:
            continue
    return None

# ---- Load & parse ----
try:
    raw = load_any(FILE_PATH)
except Exception as e:
    st.error("❌ Failed to read the file.")
    with st.expander("Show technical details"):
        st.exception(e)
    st.stop()

if raw.empty:
    st.error("The file is empty.")
    st.stop()

date_col = detect_date_column(raw)
if date_col is None:
    st.error("Could not detect a date column automatically. "
             "Please rename your date column to 'Date' or one of: day, timestamp, ts, trade_date.")
    st.stop()

# Parse to datetime and drop invalid rows
raw = raw.copy()
raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
raw = raw.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)

st.success(f"Loaded {len(raw):,} rows • Detected date column: **{date_col}**")

# ---- Build list of available dates ----
available_dates = sorted(raw[date_col].dt.date.unique())
if not available_dates:
    st.error("No valid date values found after parsing.")
    st.stop()

min_day, max_day = available_dates[0], available_dates[-1]

# ---- UI: pick a date (limit to available range) ----
col1, col2 = st.columns([1,1])
with col1:
    picked = st.date_input("Pick a date from your data", value=max_day, min_value=min_day, max_value=max_day)
with col2:
    picked_dropdown = st.selectbox("…or choose from available dates", options=available_dates, index=len(available_dates)-1)

selected_day = picked if (min_day <= picked <= max_day) else picked_dropdown

st.markdown("---")

# ---- Filter and show rows ----
rows = raw[raw[date_col].dt.date == selected_day]
if rows.empty:
    st.warning(f"No rows on **{selected_day}**. Try another date.")
else:
    st.subheader(f"🧾 Rows for {selected_day}")
    st.dataframe(rows, use_container_width=True)

    # Quick numeric summary (optional)
    num_cols = rows.select_dtypes("number").columns.tolist()
    if num_cols:
        st.markdown("**Summary (numeric columns):**")
        st.dataframe(rows[num_cols].describe().T, use_container_width=True)

# ---- Extra convenience ----
with st.expander("Show first & last few rows in the file"):
    st.write("**Head (5):**")
    st.dataframe(raw.head(), use_container_width=True)
    st.write("**Tail (5):**")
    st.dataframe(raw.tail(), use_container_width=True)

"
           + ", ".join(CANDIDATE_FILES) + ". The app auto-detects the date column; no lags or rolling used.")
'''

app_path.write_text(code, encoding="utf-8")
str(app_path)
