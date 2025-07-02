from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# === Load Models ===
detection_model = joblib.load('model/diabetes_detection_model.pkl')  # Predicts diabetic or not
type_model_bundle = joblib.load('model/diabetes_type_model.pkl')
if isinstance(type_model_bundle, dict):
    type_model = type_model_bundle['model']
else:
    type_model = type_model_bundle

# === Load Encoders ===
encoders_path = 'model/encoders/'

gender_encoder = joblib.load(os.path.join(encoders_path, 'gender_encoder.pkl'))
age_encoder = joblib.load(os.path.join(encoders_path, 'age_encoder.pkl'))
insulin_encoder = joblib.load(os.path.join(encoders_path, 'insulin_encoder.pkl'))
A1C_encoder = joblib.load(os.path.join(encoders_path, 'A1Cresult_encoder.pkl'))
diabetesMed_encoder = joblib.load(os.path.join(encoders_path, 'diabetesMed_encoder.pkl'))
change_encoder = joblib.load(os.path.join(encoders_path, 'change_encoder.pkl'))
smoking_encoder = joblib.load(os.path.join(encoders_path, 'smoking_encoder.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # === Collect Input from Form (Page 1) ===
    gender = request.form['gender']
    age = request.form['age']
    hypertension = request.form['hypertension']
    heart_disease = request.form['heart_disease']
    smoking_history = request.form['smoking_history']
    bmi = float(request.form['bmi'])
    HbA1c_level = float(request.form['hba1c_level'])
    blood_glucose_level = float(request.form['blood_glucose_level'])

    # === Encode Input (update as per your model's requirements) ===
    input_data = pd.DataFrame([{
        'gender': gender_encoder.transform([gender])[0],
        'age': age_encoder.transform([age])[0],
        'hypertension': 1 if hypertension == 'Yes' else 0,
        'heart_disease': 1 if heart_disease == 'Yes' else 0,
        'smoking_history': smoking_encoder.transform([smoking_history])[0],
        'bmi': bmi,
        'HbA1c_level': HbA1c_level,
        'blood_glucose_level': blood_glucose_level
    }])

    # === First Model: Diabetic or Not ===
    diabetic_prediction = detection_model.predict(input_data)[0]

    if diabetic_prediction == 0:
        result = "The patient is not diabetic."
        result_class = "not-diabetic"
        result_icon = "fa-circle-check"
        show_next = False
    else:
        result = "The patient is diabetic. Click below to predict the type."
        result_class = "diabetic"
        result_icon = "fa-circle-exclamation"
        show_next = True

    return render_template('index.html', result_message=result, result_class=result_class, result_icon=result_icon, show_next=show_next)

@app.route('/predict_type', methods=['GET', 'POST'])
def predict_type():
    if request.method == 'GET':
        return render_template('type.html')
    
    # === Collect Input from Form (Page 2) ===
    gender = request.form['gender']
    age = request.form['age']
    time_in_hospital = int(request.form['time_in_hospital'])
    num_medications = int(request.form['num_medications'])
    num_outpatient = int(request.form['num_outpatient'])
    num_inpatient = int(request.form['num_inpatient'])
    num_emergency = int(request.form['num_emergency'])
    insulin = request.form['insulin']
    A1Cresult = request.form['a1cresult']
    diabetesMed = request.form['diabetesMed']
    change = request.form['change']    # === Encode Input ===
    input_data = pd.DataFrame({
        'gender': [gender_encoder.transform([gender])[0]],
        'age': [age_encoder.transform([age])[0]],
        'time_in_hospital': [time_in_hospital],
        'num_medications': [num_medications],
        'number_outpatient': [num_outpatient],
        'number_inpatient': [num_inpatient],
        'number_emergency': [num_emergency],
        'insulin': [insulin_encoder.transform([insulin])[0]],
        'A1Cresult': [A1C_encoder.transform([A1Cresult])[0]],
        'diabetesMed': [diabetesMed_encoder.transform([diabetesMed])[0]],
        'change': [change_encoder.transform([change])[0]]
    })

    # === Second Model: Type 1 or Type 2 ===
    diabetes_type = type_model.predict(input_data)[0]
    if diabetes_type == 1:
        result = "The patient has Type 1 diabetes."
        result_class = "type1"
        result_icon = "fa-circle-info"
    else:
        result = "The patient has Type 2 diabetes."
        result_class = "type2"
        result_icon = "fa-circle-exclamation"

    return render_template('type.html', result_message=result, result_class=result_class, result_icon=result_icon)

if __name__ == '__main__':
    app.run(debug=True)
