import zipfile
import os

MODEL_DIR = "saved_mental_status_bert"
ZIP_FILE = "saved_mental_status_bert.zip"

# Extract model if not already extracted
if not os.path.exists(MODEL_DIR):
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)