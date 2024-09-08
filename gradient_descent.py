import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Download and load the Raisin dataset from GitHub
url = "https://github.com/ztian360/Dataset/raw/main/Raisin_Dataset.xlsx"
response = requests.get(url)
data = pd.read_excel(BytesIO(response.content))

# Preprocess the Raisin dataset
# Assuming 'Class' is the target column and the rest are features
data.dropna(inplace=True)  # Remove any rows with missing values

# Convert categorical columns to numeric if needed (e.g., 'Class' is assumed to be the target)
# For binary classification (if 'Class' column contains categorical data)
data['Class'] = data['Class'].apply(lambda x: 1 if x == 'Kecimen' else 0)

# Split the dataset into features (X) and target (y)
X = data.drop('Class', axis=1)  # All columns except 'Class' are features
y = data['Class']  # 'Class' column is the target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add intercept term to X
X_scaled = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

# Split the dataset into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Gradient Descent Function
def compute_mse(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    mse = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return mse

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    mse_history = np.zeros(iterations)

    for i in range(iterations):
        gradient = (1/m) * X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate * gradient
        mse_history[i] = compute_mse(X, y, theta)

    return theta, mse_history

# Initialize theta and set hyperparameters
theta = np.zeros(X_train.shape[1])

learning_rate = 0.01
iterations = 1000

# Perform gradient descent
theta_final, mse_history = gradient_descent(X_train, y_train, theta, learning_rate, iterations)

# Plot MSE over iterations
plt.plot(range(iterations), mse_history)
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.title("MSE vs. Iterations for Raisin Dataset")
plt.show()

# Predict on test data and compute final test MSE
y_pred = X_test.dot(theta_final)
test_mse = compute_mse(X_test, y_test, theta_final)
print(f"Test MSE: {test_mse}")

# Log the parameters and MSE
with open("gradient_descent_log.txt", "w") as log_file:
    log_file.write(f"Learning Rate: {learning_rate}\n")
    log_file.write(f"Iterations: {iterations}\n")
    log_file.write(f"Final Training MSE: {mse_history[-1]}\n")
    log_file.write(f"Test MSE: {test_mse}\n")
