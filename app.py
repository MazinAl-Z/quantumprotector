# app.py

from flask import Flask, render_template, request
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel as QuantumKernel
from qiskit_machine_learning.algorithms import QSVC
import joblib

app = Flask(__name__)

# Replace 'your_file.csv' with the actual path to your CSV file
file_path = 'dataset_clean.csv'

# Load the QSVC model (replace 'your_model_path' with the actual path to your saved model)
qsvc = joblib.load('qsvc_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Get user inputs from the form as 'yes' or 'no'
    field1 = 1.0 if request.form['field1'].lower() == 'yes' else 0.0
    field2 = 1.0 if request.form['field2'].lower() == 'yes' else 0.0
    field3 = 1.0 if request.form['field3'].lower() == 'yes' else 0.0

    # Create a DataFrame with the user inputs
    data = {'field1': [field1], 'field2': [field2], 'field3': [field3]}
    df = pd.DataFrame(data)

    # Perform prediction using the loaded QSVC model
    prediction = qsvc.predict(df)

    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
