import streamlit as st
import pickle
import joblib
import pandas as pd

# Load the trained models
cardio = joblib.load('cardio_prediction_model.joblib')

def Code(original_value):
    mapping = {
    'Male': 1,
    'Female': 0,
    'Yes': 1,
    'No': 0,
    'Govt_job': 2,
    'Private': 0,
    'Self-employed': 1,
    'Children': 3,
    'Never_worked': 4,
    'Urban': 1,
    'Rural': 0,
    'never smoked': 0,
    'formerly smoked': 1,
    'smokes': 2,
    'Unknown': 3,
    'Normal':1,
    'Above normal':2,
    'Well above normal':3
}

    encoded_value = mapping.get(original_value, None)
    return encoded_value


def predict(gender, age, hypertension,
                         bmi, smoking_status,
                         cholesterol, alco, gluc_encoded,
                         active):
    # Create a DataFrame to hold the input features
    input_cardio = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'cholesterol': [cholesterol],
        'gluc': [gluc_encoded],
        'smoke': [smoking_status],
        'alco': [alco],
        'active': [active],
        'hypertension':[hypertension],
        'bmi':[bmi]
    })


    # Make predictions using all the models
    cardio_pred_prob = cardio.predict(input_cardio)


    if cardio_pred_prob[0] <0:
        cardio_pred_prob[0] = 0.00
    elif cardio_pred_prob[0]>100:
        cardio_pred_prob[0] = 1.00


    predictions = {
        'Cardio': cardio_pred_prob[0]*100
    }

    return predictions

def main():

    style = """<div style='background-color:skyblue; padding:12px'>
              <h1 style='color:black'>DISEASE PREDICTION</h1>
       </div>"""
    st.markdown(style, unsafe_allow_html=True)

    left, right, re = st.columns(3)
    gender=left.selectbox('Gender',('Male', 'Female'))
    gender_encoded = Code(gender)

    age = right.number_input('Age',
                                  step =1.0, format="%.2f", value=1.0)
    
    hypertension=left.selectbox('Hypertension',('Yes', 'No'))
    hypertension_encoded = Code(hypertension)

    bmi = left.number_input('BMI',
                                  step =1.0, format="%.2f", value=1.0)
    
    smoking_status=right.selectbox('Smoking Status',('never smoked', 'formerly smoked', 'smokes', 'Unknown'))
    smoking_status_encoded = Code(smoking_status)
    
    
    cholesterol=right.selectbox('Cholesterol',('Normal', 'Above normal', 'Well above normal'))
    cholesterol_encoded = Code(cholesterol)

    gluc=left.selectbox('Glucose',('Normal', 'Above normal', 'Well above normal'))
    gluc_encoded = Code(gluc)

    alco =right.selectbox('Alcohol intake',('Yes', 'No'))
    alco_encoded = Code(alco)

    active =right.selectbox('Physical activity',('Yes', 'No'))
    active_encoded = Code(active)

    # Add a button to make predictions
    button = st.button("Predict")

    # if button is pressed
    if button:
        
        # make prediction
        predictions = predict(gender_encoded, age, hypertension_encoded,
                         bmi, smoking_status_encoded,
                         cholesterol_encoded, 
                         alco_encoded, gluc_encoded, active_encoded)
        for x, prob in predictions.items():
            re.success(f'{x} Probability: {prob:.2f}%')  # Display the predictions for each model

if __name__ == "__main__":
    main()