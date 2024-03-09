# CPSC483 - Lab #1: Understanding Feedforward Neural Network
# Programmer: Taylor Kim
#
# Building 3 Feedforward Neural Networks to simulate function
# y = xsin(x^2/300)
# in range x: +-100

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense


#### Part 1: Data Preparation
# Generate training data for model
np.random.seed(42)
x_train = np.linspace(-100, 100, 800)  # 800 evenly spaced points
y_train = x_train * np.sin(x_train**2 / 300)


#### Part 2: Build Models
# Splitting the data, 40% for training, 60% for testing
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.6, random_state=42)

# Model 1
model1 = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),
    Dense(1)
])

# Model 2
model2 = Sequential([
    Dense(5, activation='sigmoid', input_shape=(1,)),
    Dense(10, activation='relu'),
    Dense(1)
])

# Model 3
model3 = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),
    Dense(15, activation='sigmoid'),
    Dense(1)
])

model1.compile(optimizer='adam', loss='mean_squared_error')
model2.compile(optimizer='adam', loss='mean_squared_error')
model3.compile(optimizer='adam', loss='mean_squared_error')


#### Part 3: Model Evaluation
# Train the models
model1.fit(x_train, y_train, epochs=1000, validation_split=0.4, verbose=0)
model2.fit(x_train, y_train, epochs=1000, validation_split=0.4, verbose=0)
model3.fit(x_train, y_train, epochs=1000, validation_split=0.4, verbose=0)

# Evaluate the models
eval1 = model1.evaluate(x_test, y_test)
eval2 = model2.evaluate(x_test, y_test)
eval3 = model3.evaluate(x_test, y_test)

print("Model 1 Evaluation:", eval1)
print("Model 2 Evaluation:", eval2)
print("Model 3 Evaluation:", eval3)

# Prediction Plots
prediction1 = model1.predict(x_test)
prediction2 = model2.predict(x_test)
prediction3 = model3.predict(x_test)

plt.figure(figsize=(10, 6))
plt.plot(x_train, y_train, label='Goal Function')
plt.scatter(x_test, prediction1, label='Model 1 Prediction', marker='o', s=30)
plt.scatter(x_test, prediction2, label='Model 2 Prediction', marker='x', s=30)
plt.scatter(x_test, prediction3, label='Model 3 Prediction', marker='s', s=30)
plt.legend()
plt.title('Goal Function vs Model Predictions')
plt.show()

#### Part 4: Get Model Output and Feedforward by yourself
highest_accu_model = model1 

# Extract weights and bias from the output layer
weights = highest_accu_model.layers[1].get_weights()[0]
bias = highest_accu_model.layers[1].get_weights()[1]

print("weights: ", weights)
print("bias:", bias)

# Choosing 5 data from the training dataset
sample_data = x_train[:5]


# Model output
output = np.dot(sample_data, weights) + bias
prediction = highest_accu_model.predict(sample_data)

# Compare the results: Model output vs Model prediction
print("Output:")
print(output)
print("\nModel Prediction:")
print(prediction)
