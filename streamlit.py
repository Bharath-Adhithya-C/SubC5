import streamlit as st
import pandas as pd
from joblib import load

# Load the trained models
netflix_model = load('Netflix_model.joblib')
primevideo_model = load('PrimeVideo_model.joblib')
hotstar_model = load('Hotstar_model.joblib')
zee5_model = load('Zee5_model.joblib')

st.title("OTT Churn Prediction")

# User input form
st.header("Check Churn Status")

# Upload user details from CSV file
uploaded_file = st.file_uploader("Choose a CSV file with user details", type="csv")

if uploaded_file is not None:
    # Read user details from the uploaded CSV file
    user_data = pd.read_csv(uploaded_file)

    # Display a dropdown to select a user
    selected_user = st.selectbox("Select a user:", user_data['Username'].tolist())

    # Extract the selected user's watch time data
    selected_user_data = user_data[user_data['Username'] == selected_user][['Netflix_Watch_Time', 'PrimeVideo_Watch_Time', 'Hotstar_Watch_Time', 'Zee5_Watch_Time']]

    # Make predictions using the trained models
    netflix_prediction = netflix_model.predict(selected_user_data)[0]
    primevideo_prediction = primevideo_model.predict(selected_user_data)[0]
    hotstar_prediction = hotstar_model.predict(selected_user_data)[0]
    zee5_prediction = zee5_model.predict(selected_user_data)[0]

    # Display results
    st.subheader("Churn Prediction Results")

    def display_churn_status(platform, prediction):
        if prediction == 1:
            st.write(f"{selected_user} should consider unenrolling from {platform}.")
        else:
            st.write(f"{selected_user} doesn't need to unenroll from {platform} based on watch time.")

    display_churn_status('Netflix', netflix_prediction)
    display_churn_status('Prime Video', primevideo_prediction)
    display_churn_status('Hotstar', hotstar_prediction)
    display_churn_status('Zee5', zee5_prediction)
