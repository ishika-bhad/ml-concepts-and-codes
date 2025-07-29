#linear Regression
#ex. years of experience vs salary

import numpy as np

class LinearRegression ():
    
    #initializing the hyperparameters (learning rate and number of iterations)
    def __init__(self, learning_rate, n_iterations): #hyper parameter
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations


    def fit(self, X, Y):
        # no. of training examples and features
        self.m, self.n = X.shape  #no.of rows and columns in X

        # initializing weights and bias
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # gradient descent
        for i in range(self.n_iterations):
            self.update_weights()


    def updat_weight(self): 
        Y_prediction = self.predict(self.X)  # Predicting the output using the current weights
         
        # Calculating gradients
        dw = - (2* (self.X.T).dot(self.Y - Y_prediction)) / self.m  # Gradient w.r.t weights
        db = - 2 * np.sum(self.Y - Y_prediction) / self.m  # Gradient w.r.t bias

        # Updating weights and bias
        self.W = self.W - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    #line function for prediction
    def predict(self, X):
        return X.dot(self.W) + self.b    

#using linear regression model foe prediction
#importing libraries
import pandas as pd  # Data manipulation library
from sklearn.model_selection import train_test_split # For splitting the dataset into training and test sets    
import matplotlib.pyplot as plt # Plotting library

#data preprocessing
salary_dataset = pd.read_csv('salary_data.csv')  # Load the dataset
salary_dataset.isnull.sum()

#splitting the dataset into features and target variable
x = salary_dataset.iloc[:, :-1].values  # Independent variable features
y = salary_dataset.iloc[:, 1].values  # Dependent variable target

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=2)  # Split the dataset into training and test sets
#training the model
model = LinearRegression(learning_rate=0.02, n_iterations=1000)  # Create an instance of the LinearRegression class
model.fit(X_train, Y_train)  # Fit the model to the training data

#printing the predicted values
print('Weight = ',model.W[0], ' Bias = ', model.b)  # Print the weight and bias

#pedict the values for test data
test_data_prediction = model.predict(X_test)  # Predicting the output for the test data
print('Predicted values for test data: ', test_data_prediction)  

#visulizing the results
plt.scatter(X_test, Y_test color='blue')  # Scatter plot for test data
plt.plot(X_test, test_data_prediction, color='red')  # Line plot for predicted values

