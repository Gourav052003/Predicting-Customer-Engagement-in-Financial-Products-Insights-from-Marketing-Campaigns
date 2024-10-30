import streamlit as st

from data_preparation import DataPreparer
from data_cleaning import DataCleaner
from data_transformation import DataTransformer
from model_training import ModelTrainer

# Title of the app
st.title("Training Model for Bank Term Deposits Subscription")

# File uploader for Excel and CSV files
uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["csv"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Determine file type and read accordingly
    try:
        
        data = DataPreparer(uploaded_file).getData()
        st.write("Uploaded DataFrame:")
        st.dataframe(data)
        
        cleaned_data = DataCleaner(data).cleanData()
        transformed_data = DataTransformer(cleaned_data).transform()
        results = ModelTrainer(transformed_data).train()
        
        st.write("Results: ")
        st.dataframe(results)
        
    except Exception as e:
        st.error(f"Error reading the file: {e}")

# Instructions or information
st.info("Drag and drop your Excel or CSV file into the box above or click to browse.")
