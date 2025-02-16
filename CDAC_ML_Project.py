import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import re
import nltk
from nltk.corpus import stopwords
import os
import zipfile
import pandas as pd


MODEL_DIR = "saved_mental_status_bert"
ZIP_FILE = "saved_mental_status_bert.zip"
LABEL_ENCODER_PATH = "label_encoder.pkl"
CSV_FILE = "Processed_Doctors_Info_v2.csv"

if not os.path.exists(MODEL_DIR):
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)

nltk.download('stopwords')


model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

psychiatrists_df = pd.read_csv(CSV_FILE)
psychiatrists_df.reset_index(drop=True, inplace=True)

stop_words = set(stopwords.words('english'))


def clean_statement(statement):

    statement = statement.lower()
    statement = re.sub(r'[^\w\s]', '', statement)
    statement = re.sub(r'\d+', '', statement)
    words = statement.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


def detect_anxiety(text):

    cleaned_text = clean_statement(text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=200)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return label_encoder.inverse_transform([predicted_class])[0]


def filter_psychiatrists(location):

    if location in ['Hadapsar, Pune']:
        return psychiatrists_df[psychiatrists_df['Location'] == location][
            ['Name', 'Experience', 'Fees', 'Hospital', 'Location', 'Summary', 'Profile Link']
        ]
    elif location == 'Wakad, Pune':
        return psychiatrists_df[psychiatrists_df['Location'] == location][
            ['Name', 'Experience', 'Fees', 'Location', 'Profile Link']
        ]
    else:
        return psychiatrists_df[psychiatrists_df['Location'] == location][
            ['Name', 'Experience', 'Fees', 'Hospital', 'Location', 'Summary', 'Profile Link']
        ]


def style_table(df):

    df['Profile Link'] = df['Profile Link'].apply(lambda x: f'<a href="{x}" target="_blank">View Profile</a>')


    df['Summary'] = df['Summary'].astype(str).apply(
        lambda x: f'<span title="{x}">{x[:50]}...</span>' if isinstance(x, str) and len(x) > 50 else x
    )

    table_html = df.to_html(index=False, escape=False, classes="styled-table")

    st.markdown("""
    <style>
        .styled-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            text-align: left;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            line-height: 1.2;
        }
        .styled-table th, .styled-table td {
            padding: 6px 8px;
            border-bottom: 1px solid #ddd;
            white-space: nowrap;
        }
        .styled-table th {
            background-color: #009879;
            color: white;
            font-weight: bold;
        }
        .styled-table td a {
            color: #0073e6;
            text-decoration: none;
            font-weight: bold;
        }
        .styled-table td a:hover {
            text-decoration: underline;
        }
        .styled-table tr:hover {
            background-color: #f5f5f5;
        }
    </style>
    """, unsafe_allow_html=True)

    return table_html


st.markdown("<h1 style='color: darkgreen;'>🌿 Mental Health Status Detection</h1>", unsafe_allow_html=True)

st.write(
    "This model analyzes mental health-related text and predicts its category (e.g., Normal, Anxiety, Depression, Bipolar disorder, Stress, Suicidal).")

st.subheader("📝 Enter your mental health statement:")
input_text = st.text_area("Write your thoughts here...", height=150)

predefined_cities = psychiatrists_df["Location"].dropna().unique().tolist()

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<h5 style='margin-bottom: 5px;'>📍 Select Your Location</h5>", unsafe_allow_html=True)
location = st.selectbox("Choose your location:", predefined_cities, key="city_select")

st.markdown(
    """
    <style>
        .main .block-container { max-width: 80%; } 
        
        div.stButton > button:first-child {
            background-color: #01ad0e;
            width: 100%;
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            transition: 0.3s;
        }
        div.stButton > button:first-child:hover {
            background-color: #ACEEAC;
        }
    </style>
    """,
    unsafe_allow_html=True
)

if st.button("🔍 Detect Mental State"):
    if input_text.strip():
        predicted_class = detect_anxiety(input_text)
        st.markdown(f"""
            <div style='background-color: #d6fbff; color: #018794; padding: 10px; border-radius: 5px;'><h4>
                The Predicted Mental Health State is: <b>{predicted_class}</b></h4>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if predicted_class != "Normal":
            st.warning(
                "Your mental state seems to not be normal. Your mental well-being is important and you might benefit from professional support. Here are some trusted psychiatrists in your area who can provide support and guidance:")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            psychiatrists_in_city = filter_psychiatrists(location)
            if not psychiatrists_in_city.empty:
                styled_table = style_table(psychiatrists_in_city)
                st.markdown(styled_table, unsafe_allow_html=True)

            else:
                st.write("❌ Sorry, no psychiatrists found for the selected location.")
    else:
        st.error("⚠️ Please enter a valid statement.")
