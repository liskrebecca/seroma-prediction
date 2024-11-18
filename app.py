from flask import Flask, request, render_template
import joblib
import numpy as np
import os
import sys

app = Flask(__name__)

# Adjust the paths for PyInstaller
if hasattr(sys, '_MEIPASS'):
    base_path = sys._MEIPASS  # Base directory for PyInstaller bundled files
else:
    base_path = os.path.abspath(".")

# Update template folder for PyInstaller
app.template_folder = os.path.join(base_path, "templates")

# Load the model and the scaler
model = joblib.load(os.path.join(base_path, 'random_forest_model.pkl'))
scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))

# Function to convert form inputs to model input format and scale them
def preprocess_input(form_data):
    # Extract values from the form data
    bmi = float(form_data["input1"])
    htn = int(form_data["input2"])
    chemo_regimen = form_data["input3"]
    mastectomy_type = form_data["input4"]
    mx_wt = float(form_data["input5"])

    # Initialize the input array with zeros
    input_array = np.zeros(6)

    # Fill the input array according to the model's expected format
    input_array[0] = mx_wt
    input_array[1] = bmi
    input_array[2] = htn  # dem_htn_yes
    input_array[3] = 1 if chemo_regimen == 'neoadjuvant' else 0  # dem_chemo_neoadjuvant
    input_array[4] = 1 if mastectomy_type == 'ssm' else 0  # mx_type_ssm
    input_array[5] = 1 if mastectomy_type == 'nsm' else 0  # mx_type_nsm

    # Apply the scaler
    input_array_scaled = scaler.transform(input_array.reshape(1, -1))

    return input_array_scaled


@app.route('/')
def home():
    return render_template('index.html', prediction=None, form_data=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    form_data = request.form.to_dict()
    
    # Preprocess input
    input_data = preprocess_input(form_data)
    
    # Make prediction
    probability = model.predict_proba(input_data)[0][1]
    
    # Convert probability to percentage
    probability_percent = probability * 100
    
    return render_template('index.html', prediction=probability_percent, form_data=form_data)

if __name__ == "__main__":
    app.run(debug=True)
