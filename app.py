import streamlit as st
import pandas as pd
import os
from scripts.pipeline import process_lft_pipeline
from scripts.utils import plot_separate_patient_trends

# Streamlit App Config
st.set_page_config(page_title="LFT Analyzer", layout="wide")

# Title
st.title("Liver Function Test (LFT) Analyzer")

# Sidebar Upload Section
st.sidebar.header("📤 Upload LFT Report")
uploaded_file = st.sidebar.file_uploader("Upload an LFT Report (Image)", type=["png", "jpg", "jpeg"])

# Initialize variables
patient_id = None
detected_lang = None
data_path = None

if uploaded_file:
    # Save uploaded image temporarily
    temp_image_path = "temp_report.png"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process LFT report
    result = process_lft_pipeline(temp_image_path)  

    if isinstance(result, tuple):  # ✅ Ensure correct unpacking
        patient_id, detected_lang = result
    else:
        patient_id = result
        detected_lang = "en"  # Default language

    # Show success message
    st.sidebar.success(f"✅ Report processed successfully for Patient {patient_id}!")

    # 📂 Check if patient data exists
    data_path = f"data/patient_{patient_id}.csv"
    if os.path.exists(data_path):
        # 📊 Show patient data **before** plotting
        df = pd.read_csv(data_path)
        st.subheader(f"📂 Patient Records (ID: {patient_id})")
        st.dataframe(df)  # ✅ Display table before plots

        # 📊 Show trend plots **only if the table exists**
        st.subheader("📈 Test Trends Over Time")
        plot_separate_patient_trends(patient_id, detected_lang)  # ✅ Only called after patient data check

st.markdown("---")
st.markdown("**ℹ Note:** The app automatically processes and stores LFT data for detected patient IDs.")
