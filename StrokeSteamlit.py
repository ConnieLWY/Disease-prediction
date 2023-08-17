import streamlit as st
import pickle
import joblib
import pandas as pd

# Load the trained models
stroke = joblib.load('stroke_prediction_model.joblib')

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


def predict(gender, age, hypertension, heart_disease, ever_married,
                         work_type, Residence_type,
                         bmi, smoking_status, gluc_encoded,):
    # Create a DataFrame to hold the input features
    input_stroke = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [Residence_type],
        'bmi': [bmi],
        'smoking_status': [smoking_status],
        'gluc': [gluc_encoded]
    })


    # Make predictions using all the models
    stroke_pred_prob = stroke.predict(input_stroke)

    if stroke_pred_prob[0] <0:
        stroke_pred_prob[0] = 0.00
    elif stroke_pred_prob[0]>100:
        stroke_pred_prob[0] = 1.00


    predictions = {
        'Stroke': stroke_pred_prob[0]*100,
    }

    return predictions

def main():

    style = """<div style='background-color:skyblue; padding:12px'>
              <h1 style='color:black'>STROKE PREDICTION</h1>
       </div>"""
    st.markdown(style, unsafe_allow_html=True)
    left, right, re = st.columns(3)
    gender=left.selectbox('Gender',('Male', 'Female'))
    gender_encoded = Code(gender)

    age = right.number_input('Age',
                                  step =1.0, format="%.2f", value=1.0)
    
    hypertension=left.selectbox('Hypertension',('Yes', 'No'))
    hypertension_encoded = Code(hypertension)

    heart_disease=right.selectbox('Heart Disease',('Yes', 'No'))
    heart_disease_encoded = Code(heart_disease)

    ever_married=left.selectbox('Ever Married',('Yes', 'No'))
    ever_married_encoded = Code(ever_married)

    work_type=right.selectbox('Work Type',('Children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed'))
    work_type_encoded = Code(work_type)

    Residence_type=left.selectbox('Residence Type',('Rural', 'Urban'))
    Residence_type_encoded = Code(Residence_type)

    
    bmi = left.number_input('BMI',
                                  step =1.0, format="%.2f", value=1.0)
    
    smoking_status=right.selectbox('Smoking Status',('never smoked', 'formerly smoked', 'smokes', 'Unknown'))
    smoking_status_encoded = Code(smoking_status)
    

    gluc=left.selectbox('Glucose',('Normal', 'Above normal', 'Well above normal'))
    gluc_encoded = Code(gluc)


    # Add a button to make predictions
    button = st.button("Predict")

    # if button is pressed
    if button:
        
        # make prediction
        predictions = predict(gender_encoded, age, hypertension_encoded,
                         heart_disease_encoded, ever_married_encoded, work_type_encoded, Residence_type_encoded,
                         bmi, smoking_status_encoded,
                         gluc_encoded)
        for x, prob in predictions.items():
            re.success(f'{x} Probability: {prob:.2f}%')  # Display the predictions for each model

if __name__ == "__main__":
    main()