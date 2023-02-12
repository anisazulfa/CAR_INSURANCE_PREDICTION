import streamlit as st
import requests
from PIL import Image

# Load and set images in the first place
header_images = Image.open('assets/banner.png')
st.image(header_images)

# Add some information about the service
st.title("Car Insurance Prediction")
st.subheader("Just enter variabel below then click Predict button :sunglasses:")

# Create form of input
with st.form(key = "air_data_form"):
    # Create box for number input
    vehicle_year = st.radio(
        label = "Select year of vehicle manufacture",
        options = ["after 2015", "before 2015"],
    )

    annual_mileage = st.number_input(
        label = "Enter annual mileage value:",
        min_value = 0,
        max_value = 30000,
        help = "Value range from 0 to 30000"
    )

    gender = st.radio(
        label = "Select gender:",
        options = ["female", "male"],
    )

    education = st.radio(
        label = "Select education:",
        options = ["high school", "university", "none"],
    )

    children = st.radio(
        label = "Do you have children?",
        options = [0, 1],
        help = "Value 0: have no children; 1: have children"
    )

    duis = st.number_input(
        label = "How many times have you drive under influence?",
        min_value = 0,
        max_value = 10,
        help = "Value range from 0 to 10"
    )
    
    married = st.radio(
        label = "Are you married?",
        options = [0, 1],
        help = "Value 0: not married; 1: married"
    )
    
    speeding_violation = st.number_input(
        label = "How many times have you speeding violation?",
        min_value = 0,
        max_value = 50,
        help = "Value range from 0 to 50"
    )
    
    vehicle_ownership = st.radio(
        label = "Select vehicle ownership",
        options = [0, 1],
        help = "Value 0: non-owned; 1: owned"
    )

    past_accidents = st.number_input(
        label = "How many times have you had an accident?",
        min_value = 0,
        max_value = 50,
        help = "Value range from 0 to 50"
    )

    age = st.radio(
        label = "Select age:",
        options = ["16-25", "26-39", "40-64", "65+"],
    )

    credit_score = st.number_input(
        label = "Enter credit score:",
        min_value = 0.000000,
        max_value = 1.000000,
        format = "%.6f",
        help = "Continous value range from 0 to 1"
    )

    driving_experience = st.radio(
        label = "Select driving experience:",
        options = ["0-9y", "10-19y", "20-29y", "30y+"],
    )

    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "SPEEDING_VIOLATIONS": speeding_violation,
            "DUIS": duis,
            "PAST_ACCIDENTS": past_accidents,
            "ANNUAL_MILEAGE": annual_mileage,
            "CHILDREN": children,
            "MARRIED": married,
            "VEHICLE_OWNERSHIP": vehicle_ownership,
            "CREDIT_SCORE": credit_score,
            "AGE": age,
            "GENDER": gender,
            "DRIVING_EXPERIENCE": driving_experience,
            "EDUCATION": education,
            "VEHICLE_YEAR": vehicle_year
        }

        # Create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://localhost:8080/predict", json = raw_data).json()
            
        # Parse the prediction result
        if res["error_msg"] != "":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            if res["res"] != "Tidak mengeklaim.":
                st.warning("Melakukan klaim.")
            else:
                st.success("Tidak mengeklaim.")