import pandas as pd
from qiskit_machine_learning.algorithms import QSVC
from joblib import load

# Load the trained QSVC model from the saved file
qsvc_loaded = load('qsvc_model.joblib')

# Replace 'your_test_data.csv' with the actual path to your CSV file
test_data_path = 'dataset_testing.csv'

# Read CSV file into a pandas DataFrame
test_data = pd.read_csv(test_data_path)

# Extract input features (x_test_loaded) and target variable (y_test_loaded)
x_test_loaded = test_data.iloc[:, :3].values
y_test_loaded = test_data.iloc[:, -1].values

# Get the classification test score on the loaded test data
qsvc_score_loaded = qsvc_loaded.score(x_test_loaded, y_test_loaded)
print(f"QSVC classification test score : {qsvc_score_loaded}")

