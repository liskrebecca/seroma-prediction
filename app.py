from flask import Flask, request, render_template
import joblib
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the model and the scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to convert form inputs to model input format and scale them
def preprocess_input(form_data):
    # Extract values from the form data
    bmi = float(form_data["input1"])
    htn = int(form_data["input2"])
    chemo_regimen = form_data["input3"]
    hrt_regimen = form_data["input4"]
    mastectomy_type = form_data["input5"]
    mx_wt = float(form_data["input6"])

    # Initialize the input array with zeros
    input_array = np.zeros(8)

    # Fill the input array according to the model's expected format
    input_array[0] = mx_wt
    input_array[1] = 1 if hrt_regimen == 'adjuvant' else 0  # dem_hrt_adjuvant
    input_array[2] = bmi  # dem_bmi
    input_array[3] = htn  # dem_htn_yes
    input_array[4] = 1 if hrt_regimen == 'none' else 0  # dem_hrt_none
    input_array[5] = 1 if chemo_regimen == 'neoadjuvant' else 0  # dem_chemo_neoadjuvant
    input_array[6] = 1 if mastectomy_type == 'ssm' else 0  # mx_type_ssm
    input_array[7] = 1 if mastectomy_type == 'nsm' else 0  # mx_type_nsm

    # Scale the input array using the saved scaler
    input_array = scaler.transform(input_array.reshape(1, -1))

    return input_array

@app.route('/')
def home():
    return render_template('index.html', prediction=None, form_data=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    form_data = request.form.to_dict()

    # Print form data for debugging
    print("Form Data:", form_data)

    # Preprocess input
    input_data = preprocess_input(form_data)

    # Print preprocessed input for debugging
    print("Preprocessed Input:", input_data)
    
    # Make prediction
    probability = model.predict(input_data)[0][0]

    # Print raw prediction probability for debugging
    print("Raw Prediction Probability:", probability)
    
    # Convert probability to percentage
    probability_percent = probability * 100

    # Print prediction for debugging
    print("Probability Percent:", probability_percent)
    
    return render_template('index.html', prediction=probability_percent, form_data=form_data)

if __name__ == "__main__":
    app.run(debug=True)
