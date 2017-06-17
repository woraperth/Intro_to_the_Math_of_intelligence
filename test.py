# -------------------------------------------------
# == Gradient Descent for House Price Prediction ==
# Data Source: House Sales in King County, USA (https://www.kaggle.com/harlfoxem/housesalesprediction)
# Environment: Python 3.6
# -------------------------------------------------

# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

# -------------------------------------------
# Data checking
# -------------------------------------------
# We will choose 1 column to predict price in this step.
# -------------------------------------------
# Read data
# mydata = pd.read_csv('kc_house_data.csv')

# Get column names
# print(list(mydata))
## Output >> ['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']

# Calculate correlation between each column to price (find the most correlated one = most likely to have accurate prediction)
# print(mydata.corr())
## Output >> `sqft_living` is most correlated to price (0.7), so we will use this column to predict price.
# -------------------------------------------
# END Data checking
# -------------------------------------------

# -----------------------------------------------------
# Gradient Descent code by Matt Nedrich and Siraj Raval
# -----------------------------------------------------
# Instead of having separate variables m and b,
# One variable called `weight` is created, which consisted of 2 columns (m and b)
# Therefore, in predict_price function, we used matrix multiplication
# -----------------------------------------------------

def compute_error(this_weight, data):
    # Calculate Error
    predict = predict_price(this_weight, data['x'])
    sumError = np.sum( (data['y'] - predict)**2 ) # sum of squared error
    error = sumError / float(len(data['y'])) # divide by no. of row
    return error

def predict_price(this_weight, data):
    return np.dot(data, this_weight)

def gradient_descent_runner(this_weight, data, learning_rate):
    # Calculate slope
    N = float(len(data['y'])) # no. of row
    C = float(data['x'].shape[1]) # no. of predicting variables
    # Definition: Slope = (2/N) * (target_y - predict_y) * (-x)
    error = data['y'] - predict_price(this_weight, data['x'])
    slope = -(2/N) * np.repeat(error, repeats=C).reshape(N, C) * data['x'] # Transform error into matrix
    # Calculate new weight
    new_weight = this_weight - (learning_rate * np.sum(slope))
    return new_weight

def run():
    #Initialize variables
    learning_rate = 0.01 # Learning rate
    num_iterations = 100 # Number of iteration
    num_data = 1000 # Number of data

    # Load data and select 2 columns
    mydata = pd.read_csv('kc_house_data.csv')
    predict_data = mydata[['sqft_living', 'price']]
    # Subset data
    predict_data = predict_data.iloc[:num_data,]
    # Add intercept column
    predict_data['intercept'] = 1
    # Transform inputs to Matrix & Scale input to make prediction easier
    predict_datamat = {}
    predict_datamat['x'] = scale(predict_data.as_matrix(columns=['intercept', 'sqft_living']))
    predict_datamat['y'] = scale(predict_data['price'].as_matrix())

    # Record weight update (slope m and intercept b) in `weight
    weight = np.zeros(shape=(num_iterations, predict_datamat['x'].shape[1])) # why weight has 2 columns ? For slope and intercept
    weight[0] = np.random.rand(1, weight.shape[1]) # Random with shape (1, number of column in weight)

    print ("Starting gradient descent at weight = {0}, error = {1}".format(weight[0], compute_error(weight[0], predict_datamat)))
    print ("Running...")

    # Run Gradient Descent
    for i in range(1, num_iterations):
        weight[i] = gradient_descent_runner(weight[i - 1], predict_datamat, learning_rate)
        print('Iteration', i, weight[i], compute_error( weight[i], predict_datamat ))
    print ("After {0} iterations weight = {1}, error = {2}".format(num_iterations, weight[num_iterations - 1], compute_error(weight[num_iterations - 1], predict_datamat)))

if __name__ == '__main__':
    run()