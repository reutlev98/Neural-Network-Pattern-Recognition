import json
import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Feedforward prediction
def predict(input_data, weights):
    for layer_weights in weights:
        input_data = sigmoid(np.dot(np.array(layer_weights), np.array(input_data)))
    return np.round(input_data)

# Read the test data from "testnet1.txt"
test_data = []
with open("testnet1.txt", "r") as file:
    for line in file:
        test_data.append(line.strip())

# Read the parameters from "wnet1.json"
with open("wnet1.json", "r") as file:
    wnet1 = json.load(file)

weights = wnet1["best_solution"]
structure = wnet1["structure"]

# Perform predictions on the test data
predictions = []
for data in test_data:
    input_data = list(map(int, data))
    prediction = predict(input_data, weights)
    predictions.append(int(prediction))

# Write the predictions to "output1.txt"
with open("output1.txt", "w") as file:
    for prediction in predictions:
        file.write(str(prediction) + "\n")
