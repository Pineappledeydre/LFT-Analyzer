import os
import pandas as pd
from scripts.utils import (
    extract_text_from_image,  
    extract_lft_data,
    format_results_for_display,
    update_patient_record,
    plot_separate_patient_trends,
    preprocess_image
)

def process_lft_pipeline(image_path: str) -> str:
    """
    Complete pipeline for processing LFT reports:
    - Extract text & detect language
    - Parse details (convert Russian to English for storage)
    - Save/update patient test history
    - Plot trends (use Russian labels if detected_lang == 'ru')

    Returns:
        str: Extracted patient ID (or "12345" if missing)
    """
    print("🔄 Extracting text from LFT report...")
    extracted_text, detected_lang = extract_text_from_image(image_path)

    print("🔎 Parsing extracted text...")
    parsed_data = extract_lft_data(extracted_text, detected_lang)  # ✅ FIXED

    if not parsed_data or "patient_id" not in parsed_data or not parsed_data["patient_id"]:
        print("⚠ No patient ID found. Using default: 12345")
        patient_id = "12345"
    else:
        patient_id = str(parsed_data["patient_id"])  # ✅ Store extracted ID

    print("🛠 Formatting extracted data...")
    parsed_data["test_results"] = format_results_for_display(parsed_data["test_results"])

    print("📊 Updating patient records...")
    update_patient_record(parsed_data)  # ✅ Data is always stored in English

    print(f"📈 Visualizing NAFLD trend for Patient {patient_id}...")
    plot_separate_patient_trends(patient_id, detected_lang)  # ✅ Use Russian labels if detected

    return patient_id  # ✅ Return extracted or fallback ID
