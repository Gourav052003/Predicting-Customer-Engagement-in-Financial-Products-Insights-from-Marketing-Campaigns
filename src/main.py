import streamlit as st
from data_preparation import DataPreparer
from data_cleaning import DataCleaner
from data_transformation import DataTransformer
from data_prediction import DataPrediction

# Title of the app
st.title("Customer Engagement Predictor for Finanacial Products")

# Create two columns
col1, col2 = st.columns(2)

# Integer input fields in the first column
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    balance = st.number_input("Balance", min_value=-100000, value=0)
    day = st.number_input("Day", min_value=1, max_value=31, value=1)
    duration = st.number_input("Duration (in seconds)", min_value=0, value=60)
    campaign = st.number_input("Campaign", min_value=1, value=1)

# Integer input fields in the second column
with col2:
    pdays = st.number_input("Previous Days", min_value=-1, value=-1)  # -1 indicates not previously contacted
    previous = st.number_input("Previous Contacts", min_value=0, value=0)

# Dropdown fields in the first column
with col1:
    job = st.selectbox("Job", [
        'management', 'technician', 'entrepreneur', 'blue-collar',
        'unknown', 'retired', 'admin.', 'services', 'self-employed',
        'unemployed', 'housemaid', 'student'
    ])
    
    marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
    education = st.selectbox("Education Level", ['tertiary', 'secondary', 'unknown', 'primary'])

# Dropdown fields in the second column
with col2:
    default = st.selectbox("Default Credit", ['no', 'yes'])
    housing = st.selectbox("Housing Loan", ['yes', 'no'])
    loan = st.selectbox("Personal Loan", ['no', 'yes'])

# Additional dropdowns for contact info in the first column
with col1:
    contact = st.selectbox("Contact Communication Type", ['unknown', 'cellular', 'telephone'])
    month = st.selectbox("Last Contact Month", [
         'jan','feb','mar', 'apr','may', 'jun', 'jul', 'aug', 'sep','oct', 'nov', 'dec'
    ])

# Poutcome dropdown in the second column
with col2:
    poutcome = st.selectbox("Previous Outcome", ['unknown', 'failure', 'other', 'success'])

# Submit button
if st.button("Submit"):
    data = DataPreparer(age, balance, day, duration, campaign, 
                 pdays, previous, job, marital, education, 
                 default, housing, loan, contact, month, poutcome).getData()
    
    st.write(data)
    cleaned_data = DataCleaner(data).cleanData()
    st.write(cleaned_data)
    transformed_data = DataTransformer(cleaned_data).transform()
    st.write(transformed_data)
    result = DataPrediction(transformed_data).predict()
    
    if result[0]==0:
        st.markdown("<h4 style='color: red;'>No, Customer will not be subscribing to the Bank-Term Deposits</h4>", unsafe_allow_html=True)
    else:
        st.markdown("<h4 style='color: green;'>Yes, Customer will be subscribing to the Bank-Term Deposits</h4>", unsafe_allow_html=True)