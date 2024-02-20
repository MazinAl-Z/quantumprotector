import pandas as pd
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel as QuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from sklearn import datasets

# Replace 'your_file.csv' with the actual path to your CSV file
file_path = 'dataset_clean.csv'

# Read CSV file into a pandas DataFrame
db = pd.read_csv(file_path)

# Extract input features (x_db) and target variable (y_db)
x_db = db.iloc[:, :3].values  # Convert to NumPy matrix
y_db = db.iloc[:, -1].values  # Convert to NumPy matrix

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

# Testing
qsvc_score = qsvc.score(x_test, y_test)
print(f"QSVC classification test score: {qsvc_score}")
