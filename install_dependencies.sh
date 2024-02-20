#!/bin/bash

# Update the package list
sudo apt update

# Install necessary packages
sudo apt install -y python3 python3-pip python3-venv build-essential

# Install Qiskit dependencies
sudo apt install -y libblas-dev liblapack-dev libatlas-base-dev cython

# Install Qiskit
pip3 install qiskit

# Install other project dependencies
pip3 install pandas numpy scikit-learn qiskit_machine_learning

# Optionally, install any other dependencies your project may need

echo "Dependencies installation completed."

