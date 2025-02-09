import os
import re
import numpy as np
import pandas as pd
from fuzzywuzzy import process
import pytesseract
import langdetect
from deep_translator import GoogleTranslator
from fastapi import FastAPI, File, UploadFile
import cv2
import pytesseract
import matplotlib.pyplot as plt
import io
import streamlit as st  
import matplotlib.pyplot as plt

# Set Tesseract path manually for Streamlit Cloud
if "TESSDATA_PREFIX" not in os.environ:
    os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata/"
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Define standard column mappings
STANDARD_COLUMNS = {
    "Total Bilirubin": "Total Bilirubin",
    "Direct Bilirubin": "Direct Bilirubin",
    "Indirect Bilirubin": "Indirect Bilirubin",
    "Alanine Aminotransferase (ALT/SGPT)": "Alanine Aminotransferase (ALT/SGPT)",
    "Aspartate Aminotransferase (AST/SGOT)": "Aspartate Aminotransferase (AST/SGOT)",
    "Alkaline Phosphatase (ALP)": "Alkaline Phosphatase (ALP)",
    "Gamma-Glutamyl Transferase (GGT)": "Gamma-Glutamyl Transferase (GGT)",
    "Total Protein": "Total Protein",
    "Albumin": "Albumin",
    "Globulin": "Globulin",
    "Albumin/Globulin (A/G) Ratio": "Albumin/Globulin (A/G) Ratio",
    "Prothrombin Time (PT)": "Prothrombin Time (PT)",
    "International Normalized Ratio (INR)": "International Normalized Ratio (INR)"
}

RUSSIAN_LABELS = {
    "Total Bilirubin": "Общий билирубин",
    "Direct Bilirubin": "Прямой билирубин",
    "Indirect Bilirubin": "Непрямой билирубин",
    "Alanine Aminotransferase (ALT/SGPT)": "АЛТ (аланинаминотрансфераза)",
    "Aspartate Aminotransferase (AST/SGOT)": "АСТ (аспартатаминотрансфераза)",
    "Alkaline Phosphatase (ALP)": "Щелочная фосфатаза",
    "Gamma-Glutamyl Transferase (GGT)": "Гамма-глутамилтрансфераза",
    "Total Protein": "Общий белок",
    "Albumin": "Альбумин",
    "Globulin": "Глобулин",
    "Albumin/Globulin (A/G) Ratio": "Соотношение альбумин/глобулин",
    "Prothrombin Time (PT)": "Протромбиновое время",
    "International Normalized Ratio (INR)": "Международное нормализованное отношение"
}

REFERENCE_RANGES = {
    "Total Bilirubin": (0.5, 20.5),
    "Direct Bilirubin": (0.1, 5),
    "Indirect Bilirubin": (0.1, 16.4),
    "Alanine Aminotransferase (ALT/SGPT)": (0.1, 45),
    "Aspartate Aminotransferase (AST/SGOT)": (0.1, 45),
    "Alkaline Phosphatase (ALP)": (40,400),
    "Gamma-Glutamyl Transferase (GGT)": (0,55),
    "Total Protein": (62,83),
    "Albumin": (32,54),
    "Globulin": (13,77),
    "Albumin/Globulin (A/G) Ratio": (1.5,2.4),
    "Prothrombin Time (PT)": (9, 12.5),
    "International Normalized Ratio (INR)": (0.7, 1.5)
}

def detect_language(text: str) -> str:
    """
    Detects the language of a given text.
    Returns 'en' for English, 'ru' for Russian, or 'unknown' if detection fails.
    """
    try:
        return langdetect.detect(text)
    except langdetect.lang_detect_exception.LangDetectException:
        return "unknown"

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocesses an LFT image for better OCR digit extraction.
    - Converts to grayscale
    - Enhances contrast using CLAHE
    - Sharpens image using kernel
    - Uses Gaussian Blur for denoising
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Error: The file '{image_path}' does not exist or cannot be read.")
    # to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE for better contrast (avoids overexposure)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)
    # Apply Sharpening Kernel to enhance digits
    sharpening_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, sharpening_kernel)
    # Apply Gaussian Blur to reduce noise while keeping digits readable
    processed_img = cv2.GaussianBlur(sharpened, (3,3), 0)
    return processed_img

def extract_text_from_image(image_path: str):
    """
    Extracts text from an LFT medical image using OCR and detects its language.
    - Uses `image_to_string()` for full-text extraction.
    - Detects and translates Russian text if needed.
    """
    processed_img = preprocess_image(image_path)

    extracted_text = pytesseract.image_to_string(processed_img, lang="rus+eng")

    # Detect language
    try:
        detected_lang = langdetect.detect(extracted_text)
    except langdetect.lang_detect_exception.LangDetectException:
        detected_lang = "unknown"

    print(f"📝 Extracted Test Text: {extracted_text[:500]}") 
    if detected_lang == "ru":
        print("🔄 Translating Russian text to English...")
        extracted_text = GoogleTranslator(source='ru', target='en').translate(extracted_text[:5000])

    return extracted_text, detected_lang

def extract_lft_data(text: str, detected_lang: str) -> dict:
    """
    Extracts patient and test details from the LFT report.
    - Uses language detection to properly match field names (English or Russian).
    - Fixes date extraction for both English & Russian formats.
    """
    # English & Russian field mappings
    field_patterns = {
        "patient_name": r"(Patient Name|Имя пациента)[:\s]+([A-Za-zА-Яа-я\s]+)",
        "patient_id": r"(Patient ID|ID пациента)[:\s]+([\d]+)",
        "age": r"(Age|Возраст)[:\s]+([\d]+)",
        "gender": r"(Gender|Пол)[:\s]+(Male|Female|Мужской|Женский)",
        "date_of_test": r"(Date of Test|Дата анализа)[:\s]+([\d]{2}/[\d]{2}/[\d]{4})",
        "fasting_status": r"(Fasting Status|Статус натощак)[:\s]+(Yes|No|Да|Нет)",
    }

    extracted_data = {}
    for field, pattern in field_patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            extracted_data[field] = matches[0][-1].strip()

    if detected_lang == "ru":
        extracted_data["gender"] = "Male" if extracted_data.get("gender") == "Мужской" else "Female"
        extracted_data["fasting_status"] = "Yes" if extracted_data.get("fasting_status") == "Да" else "No"

    if "date_of_test" in extracted_data and extracted_data["date_of_test"]:
        extracted_data["date_of_test"] = pd.to_datetime(
            extracted_data["date_of_test"], format="%d/%m/%Y", errors="coerce"
        ).strftime("%Y-%m-%d")

    test_results = []
    test_pattern = r"([\w\s()/,-]+?)\s+([\d.]+)\s+([\d.]+[-–]\s*[\d.]+)?\s+([\w/%]+)"

    matches = re.findall(test_pattern, text, re.DOTALL)
    for match in matches:
        test_name = match[0].strip()
        patient_result = float(match[1]) if match[1].replace(".", "", 1).isdigit() else "N/A"
        reference_range = match[2].strip() if match[2] else "Unknown"
        units = match[3].strip()

        test_results.append({
            "test_name": test_name,
            "patient_result": patient_result,
            "reference_range": reference_range,
            "units": units
        })

    extracted_data["test_results"] = test_results

    return extracted_data

def format_results_for_display(test_results):
    """
    Formats test results to ensure correct decimal placement and removes '≈' symbols.
    - Uses expected value ranges to detect misformatted numbers.
    """
    expected_ranges = {
        "Total Bilirubin": (0.1, 5.0),
        "Direct Bilirubin": (0.0, 1.5),
        "Indirect Bilirubin": (0.1, 5.0),
        "Alanine Aminotransferase (ALT/SGPT)": (5, 500),
        "Aspartate Aminotransferase (AST/SGOT)": (5, 500),
        "Alkaline Phosphatase (ALP)": (20, 300),
        "Gamma-Glutamyl Transferase (GGT)": (5, 100),
        "Total Protein": (5.0, 9.0),
        "Albumin": (2.0, 5.5),
        "Globulin": (1.0, 4.5),
        "Albumin/Globulin (A/G) Ratio": (0.5, 3.0),
        "Prothrombin Time (PT)": (10, 20),
        "International Normalized Ratio (INR)": (0.7, 2.0)
    }

    formatted_results = []
    
    for result in test_results:
        test_name = result["test_name"]
        patient_result = result["patient_result"]
        reference_range = result.get("reference_range", "Unknown") 
        units = result["units"]

        # Apply boundary corrections
        if test_name in expected_ranges:
            min_val, max_val = expected_ranges[test_name]
            if patient_result > max_val * 10:  
                patient_result /= 100  
            elif patient_result > max_val * 5:  
                patient_result /= 10  

        # Ensure values are displayed properly without '≈'
        if isinstance(patient_result, (float, int)):
            patient_result = round(patient_result, 2)  

        formatted_results.append({
            "test_name": test_name,
            "patient_result": patient_result,  
            "reference_range": reference_range,
            "units": units
        })

    return formatted_results

def match_column_name(extracted_name):
    """
    Matches an extracted test name to the closest standard column name using fuzzy matching.
    - Returns None if no good match is found.
    """
    best_match, score = process.extractOne(extracted_name, STANDARD_COLUMNS.keys())
    return STANDARD_COLUMNS[best_match] if score > 80 else None  # Only match if confidence > 80%

def update_patient_record(data: dict):
    """
    Stores each patient's LFT results in a separate CSV file.
    - Ensures English column names even if report was in Russian.
    """
    patient_id = str(data["patient_id"])
    patient_csv_path = f"patient_{patient_id}.csv"

    # Convert test names to standard English
    test_results_dict = {
        match_column_name(test["test_name"]): test["patient_result"]
        for test in data["test_results"]
    }

    test_results_dict = {k: v for k, v in test_results_dict.items() if k is not None}  # Remove unmatched tests

    structured_data = {
        "patient_id": patient_id,
        "date_of_test": pd.to_datetime(data["date_of_test"], errors="coerce").strftime("%Y-%m-%d"),
        "age": data["age"],
        "gender": data["gender"],
        "fasting_status": data["fasting_status"]
    }
    structured_data.update(test_results_dict)

    new_entry = pd.DataFrame([structured_data])

    if os.path.exists(patient_csv_path):
        df = pd.read_csv(patient_csv_path)

        df["date_of_test"] = pd.to_datetime(df["date_of_test"], errors="coerce").dt.strftime("%Y-%m-%d")
        new_entry["date_of_test"] = pd.to_datetime(new_entry["date_of_test"], errors="coerce").dt.strftime("%Y-%m-%d")

        if structured_data["date_of_test"] not in df["date_of_test"].values:
            df = pd.concat([df, new_entry], ignore_index=True)
            df = df[["patient_id", "date_of_test", "age", "gender", "fasting_status"] + list(STANDARD_COLUMNS.values())]
            df.to_csv(patient_csv_path, index=False)
            print(f"New test added for Patient {patient_id} in `{patient_csv_path}`.")
        else:
            print(f"Test from {structured_data['date_of_test']} already exists for Patient {patient_id}.")
    else:
        new_entry = new_entry[["patient_id", "date_of_test", "age", "gender", "fasting_status"] + list(STANDARD_COLUMNS.values())]
        new_entry.to_csv(patient_csv_path, index=False)
        print(f"Created new record file `{patient_csv_path}` for Patient {patient_id}.")


def plot_separate_patient_trends(patient_id: str, language: str = "en"):
    """
    Plots selected test trends with:
    - Green corridors for normal reference ranges
    - Red shading for abnormal values
    - User selection of tests to display (Unique Key Fix Applied)
    """
    if isinstance(patient_id, tuple):  # ✅ Fix tuple issue
        patient_id = patient_id[0]

    patient_csv_path = f"data/patient_{patient_id}.csv"

    if not os.path.exists(patient_csv_path):
        st.warning(f"⚠ No records found for Patient {patient_id}.")
        return

    df = pd.read_csv(patient_csv_path)
    df["date_of_test"] = pd.to_datetime(df["date_of_test"], errors="coerce")
    df = df.sort_values(by="date_of_test")

    exclude_columns = {"patient_id", "date_of_test", "age", "gender", "fasting_status"}
    test_columns = [col for col in df.columns if col not in exclude_columns and df[col].dtype in [np.float64, np.int64]]

    if not test_columns:
        st.warning(f"⚠ No numerical test results found for Patient {patient_id}.")
        return

    # ✅ Generate a unique key using session state
    widget_key = f"test_select_{patient_id}_{st.session_state.get('run_id', 0)}"

    selected_tests = st.multiselect(
        "📊 Select Tests to Display",
        test_columns,
        default=test_columns[:3],  # Show first 3 by default
        key=widget_key  # ✅ Ensuring unique widget key
    )

    for test in selected_tests:
        plt.figure(figsize=(10, 5))

        # ✅ Plot the normal range as a green corridor
        if test in REFERENCE_RANGES:
            lower_bound, upper_bound = REFERENCE_RANGES[test]
            plt.fill_between(df["date_of_test"], lower_bound, upper_bound, color="green", alpha=0.2, label="Normal Range")

        # ✅ Highlight abnormal values (red shading)
        above_norm = df[test] > REFERENCE_RANGES.get(test, (0, np.inf))[1]
        below_norm = df[test] < REFERENCE_RANGES.get(test, (-np.inf, np.inf))[0]

        plt.plot(df["date_of_test"], df[test], marker="o", linestyle="-", label=test, color="blue")

        if above_norm.any():
            plt.fill_between(df["date_of_test"], df[test], REFERENCE_RANGES.get(test, (0, np.inf))[1], 
                             color="red", alpha=0.3, label="Above Normal")
        if below_norm.any():
            plt.fill_between(df["date_of_test"], df[test], REFERENCE_RANGES.get(test, (-np.inf, np.inf))[0], 
                             color="red", alpha=0.3, label="Below Normal")

        plt.xlabel("Test Date")
        plt.ylabel(f"{test} (U/L or mg/dL)")
        plt.title(f"Trend of {test} for Patient {patient_id}")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)

        # ✅ Use Streamlit to display the plot
        st.pyplot(plt)
        plt.close()  # ✅ Fix memory leak issue from too many figures