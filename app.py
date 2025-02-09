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

patient_id = None
detected_lang = None
data_path = None

if uploaded_file:
    # Save uploaded image temporarily
    temp_image_path = "temp_report.png"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    result = process_lft_pipeline(temp_image_path)  

    if isinstance(result, tuple):  # Ensure it's correctly unpacked
        patient_id, detected_lang = result
    else:
        patient_id = result
        detected_lang = "en"  # Default language if not returned

    # Show success message
    st.sidebar.success(f"✅ Report processed successfully for Patient {patient_id}!")

    # 📂 Load patient data before plotting
    data_path = f"data/patient_{patient_id}.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        # ✅ Show patient records FIRST
        st.subheader(f"📂 Patient Records (ID: {patient_id})")
        st.dataframe(df)
        
        # 📊 Now show test trends (after patient data exists)
        st.subheader("📈 Test Trends Over Time")
        plot_separate_patient_trends(patient_id, detected_lang)  

st.markdown("---")
st.markdown("**ℹ Note:** The app automatically processes and stores LFT data for detected patient IDs.")