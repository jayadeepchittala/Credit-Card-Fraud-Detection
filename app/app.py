import streamlit as st
import pandas as pd
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ==========================================
# LOAD MODEL AND SCALER
# ==========================================

model = joblib.load("../models/fraud_model.pkl")
scaler = joblib.load("../models/scaler.pkl")


# ==========================================
# TITLE
# ==========================================

st.title("💳 Credit Card Fraud Detection System")

st.write(
    "Upload a credit card transaction CSV file "
    "to detect potentially fraudulent transactions."
)

st.divider()


# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)


# ==========================================
# PROCESS UPLOADED FILE
# ==========================================

if uploaded_file is not None:

    # Read CSV
    data = pd.read_csv(uploaded_file)

    # Dataset Preview
    st.subheader("📊 Dataset Preview")

    st.dataframe(
        data.head(),
        use_container_width=True
    )


    # ======================================
    # REMOVE TARGET COLUMN IF PRESENT
    # ======================================

    if "Class" in data.columns:

        X = data.drop("Class", axis=1)

    else:

        X = data.copy()


    # ======================================
    # CHECK COLUMNS
    # ======================================

    expected_columns = [
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Amount"
    ]


    missing_columns = [
        column
        for column in expected_columns
        if column not in X.columns
    ]


    if missing_columns:

        st.error(
            f"Missing columns: {missing_columns}"
        )


    else:

        # Keep correct column order
        X = X[expected_columns]


        # ==================================
        # FEATURE SCALING
        # ==================================

        X_scaled = scaler.transform(X)


        # ==================================
        # PREDICTION
        # ==================================

        predictions = model.predict(X_scaled)

        probabilities = model.predict_proba(
            X_scaled
        )[:, 1]


        # ==================================
        # CREATE RESULTS
        # ==================================

        result = data.copy()

        result["Fraud_Probability"] = probabilities

        result["Prediction"] = predictions


        # ==================================
        # RISK LEVEL
        # ==================================

        result["Risk_Level"] = result[
            "Fraud_Probability"
        ].apply(
            lambda x:
            "High Risk"
            if x >= 0.70
            else
            "Medium Risk"
            if x >= 0.30
            else
            "Low Risk"
        )


        # ==================================
        # RESULTS
        # ==================================

        st.divider()

        st.subheader("🚨 Fraud Detection Results")


        # ==================================
        # METRICS
        # ==================================

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Total Transactions",
                len(result)
            )


        with col2:

            st.metric(
                "Predicted Fraud",
                int((predictions == 1).sum())
            )


        with col3:

            st.metric(
                "Predicted Genuine",
                int((predictions == 0).sum())
            )


        # ==================================
        # RESULTS TABLE
        # ==================================

        st.subheader("🔍 Transaction Predictions")

        st.dataframe(
            result,
            use_container_width=True
        )


        # ==================================
        # RISK DISTRIBUTION
        # ==================================

        st.subheader("📈 Risk Distribution")

        risk_counts = result[
            "Risk_Level"
        ].value_counts()


        st.bar_chart(risk_counts)


        # ==================================
        # SUCCESS MESSAGE
        # ==================================

        st.success(
            "✅ Fraud detection completed successfully!"
        )