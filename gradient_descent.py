import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Function to compute the Mean Squared Error (MSE)
def compute_mse(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    mse = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return mse

# Gradient Descent Function
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    mse_history = np.zeros(iterations)

    for i in range(iterations):
        # Calculate the gradient
        gradient = (1/m) * X.T.dot(X.dot(theta) - y)
        # Update theta values
        theta = theta - learning_rate * gradient
        # Save the MSE for each iteration
        mse_history[i] = compute_mse(X, y, theta)

    return theta, mse_history

# Load the dataset
# Use your actual dataset here, but for example purposes, we'll use the Boston dataset from sklearn
from sklearn.datasets import fetch_openml
boston = fetch_openml(name="Boston", version=1)

# Convert to DataFrame
data = pd.DataFrame(data=boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

# Preprocess dataset (standardize the features and remove nulls if needed)
data.dropna(inplace=True)

X = data.drop("PRICE", axis=1).values
y = data['PRICE'].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add intercept term to X (bias term)
X_scaled = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize theta values
theta = np.zeros(X_train.shape[1])

# Set hyperparameters
learning_rate = 0.01
iterations = 1000

# Perform gradient descent
theta_final, mse_history = gradient_descent(X_train, y_train, theta, learning_rate, iterations)

# Plot MSE over iterations
plt.plot(range(iterations), mse_history)
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.title("MSE vs. Iterations")
plt.show()

# Predict on the test set and compute final test MSE
y_pred = X_test.dot(theta_final)
test_mse = compute_mse(X_test, y_test, theta_final)
print(f"Test MSE: {test_mse}")

# Log file creation
with open("gradient_descent_log.txt", "w") as log_file:
    log_file.write(f"Learning Rate: {learning_rate}\n")
    log_file.write(f"Iterations: {iterations}\n")
    log_file.write(f"Final Training MSE: {mse_history[-1]}\n")
    log_file.write(f"Test MSE: {test_mse}\n")