# **Bank Term Deposit Subscription Prediction**

## **1. Project Overview**
This project leverages machine learning to predict whether a customer will subscribe to a bank's term deposit based on data collected from direct marketing campaigns. By analyzing features such as customer demographics, previous interactions, and financial data, we aim to optimize marketing strategies for future campaigns.

This repository contains the complete pipeline from data preprocessing, feature engineering, model building, hyperparameter tuning, and model evaluation.

## **2. Dataset**
The dataset used is from a Portuguese banking institution, consisting of 41,188 instances and 20 features. It contains customer data and outcomes from direct marketing campaigns involving phone calls. Key features include:

- **Customer Attributes**: `age`, `job`, `marital`, `education`, `balance`, `housing`, `loan`, etc.
- **Contact Attributes**: `contact type (telephone, cellular)`, `last contact day`, `duration`, etc.
- **Previous Campaign Data**: `pdays`, `previous`, `poutcome` (outcome of the previous campaign).
- **Target Variable**: `subscribed` (whether the customer subscribed to a term deposit).

### **Dataset Preprocessing**
- Handled missing values using median imputation and default values for categorical features.
- Encoded categorical variables using One-Hot Encoding.
- Applied **Min-Max scaling** to normalize continuous features.

## **3. Exploratory Data Analysis (EDA)**
**Objective**: Identify key patterns and relationships between features and the target variable.

- **Correlation Matrix**: Assessed correlations between numerical features and the target variable.
- **Univariate and Bivariate Analysis**: Visualized distributions of important features (e.g., age, balance) and their relationships with the target.
- **Class Imbalance**: The dataset is highly imbalanced, with only ~11% positive class (i.e., subscribed). Addressed class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**.

## **4. Feature Engineering**
- **Interaction Features**: Created new interaction terms between `balance` and `pdays` to capture potential non-linear relationships.
- **Domain-Specific Features**: Developed features such as `contact rate per campaign` and `balance-duration ratio`.
- **Temporal Features**: Derived features based on the day of the week and time of contact to account for possible temporal effects on subscription likelihood.

## **5. Model Development**
### **Baseline Models**:
1. **Logistic Regression**: As a baseline for comparison.
2. **Decision Trees**: For interpretable predictions.

### **Advanced Models**:
1. **Random Forest**: Robust model for handling non-linear relationships and feature importance analysis.
2. **XGBoost**: Gradient boosting for better generalization and handling of imbalanced classes.
3. **CatBoost**: Evaluated due to its efficiency in handling categorical features without explicit encoding.

### **Hyperparameter Tuning**:
Utilized **GridSearchCV** and **RandomizedSearchCV** for hyperparameter optimization:
- **Random Forest**: Tuned `n_estimators`, `max_depth`, and `min_samples_split`.
- **XGBoost**: Tuned `learning_rate`, `max_depth`, `n_estimators`, and `subsample`.

### **Model Evaluation Metrics**:
- **Accuracy**: Simple baseline comparison.
- **Precision**: Focus on minimizing false positives in this business context.
- **Recall**: Important to avoid missing potential customers likely to subscribe.
- **F1-Score**: Harmonic mean of precision and recall to balance both.
- **ROC-AUC Score**: Evaluated the model's ability to discriminate between the classes.

### **Handling Imbalance**:
- Implemented **SMOTE** to oversample the minority class and improve recall.
- Tested **class weights adjustment** to further balance precision and recall.

## **6. Results and Insights**
- The best-performing model was **XGBoost**, achieving:
  - **Accuracy**: 90.5%
  - **Precision**: 75.6%
  - **Recall**: 68.3%
  - **F1-Score**: 71.8%
  - **ROC-AUC**: 92.2%

- **Feature Importance** (from XGBoost):
  1. `duration`: The duration of the last contact.
  2. `pdays`: Number of days since the client was last contacted.
  3. `balance`: Customer's account balance.
  4. `campaign`: Number of contacts during the current campaign.
  5. `job`: Customer’s occupation.

- The **duration of the last contact** was the most influential predictor, indicating the importance of engagement time in a successful subscription.

## **7. Deployment and Next Steps**
### **Model Deployment**:
- The final model is deployed via **Flask API**. It accepts customer data as input and returns the likelihood of subscription.
- **Dockerized the API** for easy integration with other banking systems.

### **Potential Improvements**:
1. Experiment with **neural networks** to capture more complex patterns in high-dimensional data.
2. Integrate **real-time data** to make the model adaptive to changing customer behaviors and market trends.
3. Implement an **A/B testing framework** to continuously validate and improve the model in production.

## **8. Repository Structure**
```bash
├── data/                     # Dataset and data processing scripts
├── notebooks/                # Jupyter notebooks for EDA and model building
├── models/                   # Saved models and model training scripts
├── app/                      # Flask app for deployment
├── Dockerfile                # Docker configuration
├── README.md                 # Project documentation
└── requirements.txt          # List of dependencies
```

## **9. How to Run the Project**
Clone the repository:
```
git clone https://github.com/your-username/bank-term-deposit-prediction.git
cd bank-term-deposit-prediction
```

Install dependencies:
```
pip install -r requirements.txt
```
Run the Jupyter notebook to train models:
```
jupyter notebook notebooks/Bank_Term_Deposit_Prediction.ipynb
```
Run the Flask API for predictions:
```
cd app
python app.py
```

### Key Technical Enhancements:
- **Detailed descriptions of models, algorithms, and hyperparameter tuning techniques**.
- **Emphasis on dealing with class imbalance** using methods like SMOTE and class weight adjustments.
- **Feature engineering** techniques that demonstrate data-driven decision-making.
- **Comprehensive evaluation metrics** showing performance beyond just accuracy, including precision, recall, F1-score, and ROC-AUC.
- **Future work** that hints at more complex methods (e.g., neural networks, real-time predictions) and production considerations (e.g., Dockerization, API deployment).
