import streamlit as st
import pandas as pd
import os
from scripts.pipeline import process_lft_pipeline
from scripts.utils import plot_separate_patient_trends  # ✅ Removed detect_language (not needed)

# Streamlit App Config
st.set_page_config(page_title="LFT Analyzer", layout="wide")

# Title
st.title("Liver Function Test (LFT) Analyzer")

# Upload Section
st.sidebar.header("Upload LFT Report")
uploaded_file = st.sidebar.file_uploader("Upload an LFT Report (Image)", type=["png", "jpg", "jpeg"])

# Initialize Patient ID (fallback)
patient_id = "12345"
detected_lang = "en"

if uploaded_file:
    # Save uploaded image temporarily
    temp_image_path = "temp_report.png"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the LFT report and get patient ID + detected language
    st.sidebar.text("🔄 Processing...")
    patient_id, detected_lang = process_lft_pipeline(temp_image_path)  # FIXED

    # Show success message
    st.sidebar.success(f"Report processed successfully for Patient {patient_id}!")

# Display Patient Data
st.subheader(f"📂 Patient Records (ID: {patient_id})")

# Define patient file path dynamically
data_path = f"data/patient_{patient_id}.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    st.dataframe(df)

    # Show trend plots
    st.subheader("📈 Test Trends Over Time")
    plot_separate_patient_trends(patient_id, detected_lang)  # FIXED
else:
    st.warning(f"⚠ No records found for Patient {patient_id}. Please upload an LFT report.")

st.markdown("---")
st.markdown("**Note:** The app automatically processes and stores LFT data for detected patient IDs.")
