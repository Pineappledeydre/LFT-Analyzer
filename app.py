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

# Only show table and plots after processing
patient_id = None
detected_lang = None
data_path = None

if uploaded_file:
    st.sidebar.text("🔄 Processing...")
    patient_id = process_lft_pipeline(temp_image_path)

    # Show success message
    st.sidebar.success(f"✅ Report processed successfully for Patient {patient_id}!")

    # 📊 Show trend plots (even if table is hidden)
    st.subheader("📈 Test Trends Over Time")
    plot_separate_patient_trends(patient_id, "en")  # ✅ Call function

    # Show patient data only if available and button clicked
    data_path = f"data/patient_{patient_id}.csv"
    if os.path.exists(data_path):
        if st.button("📂 Show Patient Records"):  
            df = pd.read_csv(data_path)
            st.subheader(f"📂 Patient Records (ID: {patient_id})")
            st.dataframe(df)


st.markdown("---")
st.markdown("**ℹ Note:** The app automatically processes and stores LFT data for detected patient IDs.")
