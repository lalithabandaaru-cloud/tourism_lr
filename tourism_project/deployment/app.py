import streamlit as st
import pandas as pd
import joblib
import os

# Define paths for the model and preprocessor
MODEL_PATH = "tourism_model_v1.joblib" # Model is saved at the root of the project by model_training.py
PREPROCESSOR_PATH = "tourism_project/model_building/preprocessor.joblib" # Assuming preprocessor is saved here

# --- Load Model and Preprocessor ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved correctly.")
        return None

@st.cache_resource
def load_preprocessor():
    try:
        # Ensure the preprocessor is also saved. This step assumes it was saved during data_preparation.py
        # If not, you might need to re-create and save it or adjust the path.
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return preprocessor
    except FileNotFoundError:
        st.error(f"Preprocessor file not found at {PREPROCESSOR_PATH}. Please ensure the preprocessor is saved correctly during data preparation.")
        return None

model = load_model()
preprocessor = load_preprocessor()

# --- Streamlit App Interface ---
st.title("Tourism Product Purchase Prediction")
st.write("Enter customer details to predict if they will purchase the Wellness Tourism Package.")

if model is not None and preprocessor is not None:
    # Input fields for features
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=90, value=30)
        typeof_contact = st.selectbox("Type of Contact", ['Self Inquiry', 'Company Invited'])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Freelancer', 'Unemployed'])
        gender = st.selectbox("Gender", ['Male', 'Female'])
        number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)

    with col2:
        preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
        marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
        number_of_trips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=5)
        passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)

    with col3:
        designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'Analyst', 'Director', 'Engineer', 'Steno', 'VP', 'President', 'Clerk'])
        monthly_income = st.number_input("Monthly Income", min_value=0.0, value=50000.0, step=1000.0)
        pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
        product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King', 'Premium'])
        number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
        duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=15)

    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        'Age': [age],
        'TypeofContact': [typeof_contact],
        'CityTier': [city_tier],
        'Occupation': [occupation],
        'Gender': [gender],
        'NumberOfPersonVisiting': [number_of_person_visiting],
        'PreferredPropertyStar': [preferred_property_star],
        'MaritalStatus': [marital_status],
        'NumberOfTrips': [number_of_trips],
        'Passport': [passport],
        'OwnCar': [own_car],
        'NumberOfChildrenVisiting': [number_of_children_visiting],
        'Designation': [designation],
        'MonthlyIncome': [monthly_income],
        'PitchSatisfactionScore': [pitch_satisfaction_score],
        'ProductPitched': [product_pitched],
        'NumberOfFollowups': [number_of_followups],
        'DurationOfPitch': [duration_of_pitch]
    })

    if st.button("Predict Purchase"): # Moved button inside the if block
        try:
            # Preprocess the input data
            processed_input = preprocessor.transform(input_data)

            # Make prediction
            prediction = model.predict(processed_input)
            prediction_proba = model.predict_proba(processed_input)[:, 1]

            st.subheader("Prediction Result:")
            if prediction[0] == 1:
                st.success(f"The customer is LIKELY to purchase the package (Probability: {prediction_proba[0]:.2f})")
            else:
                st.warning(f"The customer is UNLIKELY to purchase the package (Probability: {prediction_proba[0]:.2f})")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
else:
    st.warning("Model or preprocessor could not be loaded. Please check the file paths and ensure they are saved.")

# Note: This app assumes that the 'preprocessor.joblib' file is saved during the data_preparation step.
# You might need to add a line to save the preprocessor in data_preparation.py like:
# joblib.dump(preprocessor, 'tourism_project/model_building/preprocessor.joblib')
