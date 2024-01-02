import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer

# loan model
loaded_pipeline = joblib.load('pipeline2.pkl')

def percentage(integer_part):
# scaling 30-100 into 1-100 range to calculate loan percentage
    x = integer_part - 30
    cal = x / 0.707
    loan_percentage = round(cal)
# scaling 1-100 into 1000-10,000 to calculate loan amount
    ratio = 9000/99
    amount = ratio * (loan_percentage - 1)
    loan_amount = round(amount + 1000)
    return loan_amount

def main():
    st.title("**Loan Prediction API**")
    with st.form(key='loan_prediction_form'):
        user_info = st.text_area("Enter user information (JSON format):")

        submit_button = st.form_submit_button(label='Predict Loan Status')

        if submit_button:
            st.info("**Prediction Result**")
            try:
                additional_data = json.loads(user_info)
                additional_data_df = pd.DataFrame([additional_data])
                prediction = loaded_pipeline.predict(additional_data_df)
                xy = prediction[0]
                abc = int(xy)

                st.write(f"**Loan status** : {abc}")

                if abc >30:
                    amount = percentage(abc)
                    st.write(f"**Loan Amount** : {amount}")
                else:
                    st.write(f"**You Status is Low You are Not Elegible**")

            except Exception as e:
                st.error(f"Error predicting loan status: {str(e)}")

if __name__ == "__main__":
    main()
