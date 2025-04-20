# Developed by: SIDDHARTH S
# RegisterNumber:  212224040317

# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Here's the algorithm for your multi-output regression task in 4 concise lines:

1. **Prepare the data**: Load the dataset, select features, and split into training and test sets.
2. **Scale the features**: Apply `StandardScaler` to scale the features for better model performance.
3. **Train the model**: Use `SGDRegressor` wrapped in `MultiOutputRegressor` to train the model on the training data.
4. **Evaluate the model**: Make predictions on the test set, calculate the mean squared error (MSE), R² score, and squared errors for each sample.
## Program:
```
/*
Program to implement SGD Regressor for linear regression.

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Fetch the California Housing dataset
data = fetch_california_housing()

# Choose new features: AveRooms, AveOccup, MedInc, HouseAge, Longitude
X = data.data[:, [3, 4, 0, 1, 7]]  # Features: AveRooms, AveOccup, MedInc, HouseAge, Longitude
y = np.column_stack((data.target, data.data[:, 5]))  # Target: House Price, Occupancy

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model using SGDRegressor with MultiOutputRegressor for two targets
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = multi_output_sgd.predict(X_test_scaled)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Evaluate R² for individual targets
r2_house_price = r2_score(y_test[:, 0], y_pred[:, 0])
r2_occupancy = r2_score(y_test[:, 1], y_pred[:, 1])

print(f"R² for House Price: {r2_house_price:.2f}")
print(f"R² for Occupancy: {r2_occupancy:.2f}")


*/
```

## Output:
![simple linear regression model for predicting the marks scored]

*/

Mean Squared Error: 1.99

R² Score: 0.21

R² for House Price: 0.49

R² for Occupancy: -0.08

*/




## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
