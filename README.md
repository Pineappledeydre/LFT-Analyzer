## **LFT Analyzer**
**Automated Liver Function Test (LFT) Analysis & Trend Visualization** using **Streamlit**, **OCR (Tesseract)**, and **FastAPI**.

### **Overview**
LFT Analyzer processes **Liver Function Test (LFT) reports** from **images** using **OCR**, extracts key lab values, and visualizes trends over time.

### **Project Structure**
```
📂 LFT-Analyzer/
│── 📂 data/                      # Stores patient CSV files
│    ├── patient_12345.csv        # Data for Patient 12345
│
│── 📂 scripts/                   # Core processing functions
│    ├── pipeline.py              # Main processing functions
│    ├── utils.py                 # Helper functions (date formatting, fuzzy matching)
│
│── 📂 assets/                     # Sample images, README images
│    ├── example_report.jpg    
│
│── app.py                        # Streamlit UI
│── requirements.txt               # Python dependencies
│── README.md                      # Project documentation
│── .gitignore                     # Ignore unnecessary files
```

---

## **Features**
- **Upload an LFT Image** (JPG/PNG)
- **Extract Test Results using OCR** (Tesseract)
- **Support for English & Russian Reports**
- **Automatically Detect & Translate Russian Reports**
- **Save Patient Data as CSV**
- **Plot Individual Test Trends Over Time**
- **Simple Web UI with Streamlit**  

---

## **Quick Start**
### **Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Start the Streamlit App**

## **Usage Guide**
1. **Upload an LFT Report (Image)**  
2. **Extract & Store Data** (Automatic)  
3. **View Processed Test Results** 
4. **See Individual Test Trends Over Time**  


## ** Tech Stack**
**Python**  
**Streamlit** (UI)  
**Tesseract OCR** (Text Extraction)  
**Pandas** (Data Handling)  
**Matplotlib** (Data Visualization)  
**FuzzyWuzzy** (Fuzzy Matching for Test Names)  
**Deep-Translator** (Russian → English)  
**FastAPI (Optional)**  


**LFT Analyzer** makes medical test tracking **fast, accurate, and easy!**  
Developed for **patients, researchers, and healthcare professionals.**  
