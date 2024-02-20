#to use the database form the csv file
import pandas as pd
#to fit the data into a matrix
import numpy as np
#for the machin learning
from sklearn import datasets
#for saveing the modle
from joblib import dump

#calling the quantom kernal
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel as QuantumKernel
from qiskit_machine_learning.algorithms import QSVC


# Replace 'your_file.csv' with the actual path to your CSV file
file_path = 'dataset_small.csv'

# Read CSV file into a pandas DataFrame
db = pd.read_csv(file_path)

# Extract input features (x_db) and target variable (y_db)
x_db = db.iloc[:, :3].values  # Convert to NumPy matrix
y_db = db.iloc[:, -1].values

#spliting the data into traing and testing
train_size = int(0.6 * len(x_db))
x_train = x_db[:train_size]
y_train = y_db[:train_size]
x_test = x_db[train_size:]
y_test = y_db[train_size:]

# Training
feature_map = ZZFeatureMap(feature_dimension=3, reps=2)
qkernel = QuantumKernel(feature_map=feature_map)
qsvc = QSVC(quantum_kernel=qkernel)
qsvc.fit(x_train, y_train)

# Prediction on Training Data
predictions_train = qsvc.predict(x_train)

# Save the trained QSVC model to a file
dump(qsvc, 'qsvc_model.joblib')

# Testing
qsvc_score = qsvc.score(x_test, y_test)
print(f"QSVC classification test score: {qsvc_score}")
