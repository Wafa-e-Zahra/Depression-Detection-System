import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Set page configuration
st.set_page_config(page_title="Depression Detection App", layout="wide")

# Load the pre-trained model and tokenizer
model_path = "depression_bert_model"  
try:
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
except Exception as e:
    st.error(f"Failed to load model/tokenizer: {str(e)}")

# Sidebar for navigation
page = st.sidebar.selectbox("Navigation", ["Home"])

# App Title
st.markdown("<h1 style='text-align: center;'>Depression Detection App</h1>", unsafe_allow_html=True)

# Navigation logic
if page == "Home":
    # Home Section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<p style='font-style: italic; font-size: 20px;'>Welcome to the Depression Detection Page!</p>", unsafe_allow_html=True)
    with col2:
        st.image("DP.jpeg", caption="", width=150)

    # Input Section
    st.markdown("<h3 style='text-align: center;'>Enter your text for depression analysis</h3>", unsafe_allow_html=True)

    
    user_input = st.text_area("Enter text:", placeholder="Type here...")

    # Processing Section
    if st.button("Detect"):
        if user_input.strip():
            try:
                # Preprocess the input for the model
                inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
                outputs = model(**inputs)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=1).item()

                # Map prediction to labels
                result = "Depression detected. Please seek help." if prediction == 1 else "No depression detected. Stay positive!"
                st.success(result)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please enter some text for analysis.")
