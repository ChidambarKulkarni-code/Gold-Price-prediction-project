#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ==================== IMPORT LIBRARIES ====================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="Gold Price Forecasting (Indian Market)", layout="wide")
st.title("🇮🇳 Gold Price Forecasting – Indian Market")

st.markdown("""
This app predicts **Indian Gold Prices (INR)** using economic indicators such as Oil Prices, Silver Prices, USD/INR Exchange Rate, Sensex, and Inflation Rate.  
You can upload your Excel dataset, visualize relationships, train regression models, and make predictions interactively.
""")

# ==================== UPLOAD DATA ====================
uploaded_file = st.file_uploader("📂 Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Read Excel file (first sheet automatically)
    df = pd.read_excel(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", df.columns.tolist())

    # Handle missing values
    df = df.dropna()
    st.info("Missing values removed successfully.")

    # If Date column exists, convert to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        st.write("✅ Date column converted to datetime format.")

    # ==================== EDA SECTION ====================
    st.subheader("🔍 Exploratory Data Analysis")

    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Histogram of Numeric Variables")
        fig, ax = plt.subplots()
        numeric_df.hist(ax=ax, color="gold")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="YlGnBu")
        st.pyplot(fig)

    # ==================== MODEL TRAINING ====================
    st.subheader("⚙️ Model Training & Evaluation")

    target = st.selectbox("🎯 Select Target Variable (Y)", options=numeric_df.columns)
    features = st.multiselect(
        "📈 Select Feature Columns (X)",
        options=[c for c in numeric_df.columns if c != target]
    )

    if len(features) > 0:
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model_choice = st.selectbox(
            "Choose Regression Model", 
            ["Linear Regression", "Ridge Regression", "Lasso Regression"]
        )

        if st.button("Train Model"):
            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "Ridge Regression":
                model = Ridge(alpha=1.0)
            else:
                model = Lasso(alpha=0.1)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.success(f"✅ Model Trained Successfully: {model_choice}")
            st.write(f"**Mean Squared Error (MSE):** {mse:.3f}")
            st.write(f"**R² Score:** {r2:.3f}")

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, color="purple")
            ax.set_xlabel("Actual Gold Price (INR)")
            ax.set_ylabel("Predicted Gold Price (INR)")
            ax.set_title("Actual vs Predicted Gold Prices")
            st.pyplot(fig)

            # Save model and features in session
            st.session_state.model = model
            st.session_state.features = features

    # ==================== PREDICTION SECTION ====================
    if "model" in st.session_state:
        st.subheader("💰 Predict New Gold Price")

        st.markdown("Enter values for each economic indicator below:")

        input_data = {}
        for feat in st.session_state.features:
            # Default to mean value of feature for convenience
            default_val = float(df[feat].mean())
            input_data[feat] = st.number_input(f"{feat}", value=default_val)

        if st.button("Predict Gold Price"):
            new_df = pd.DataFrame([input_data])
            prediction = st.session_state.model.predict(new_df)[0]
            st.success(f"Predicted Gold Price (INR): **₹{prediction:,.2f}**")

else:
    st.info("👆 Please upload your Excel file (.xlsx) to begin.")

