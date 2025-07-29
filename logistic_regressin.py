import numpy as np
import pandas as pd

class LogisticRegression () :
    # Initializing the hyperparameters (learning rate, no of iterationsr)
    def __init__(self, learning_rate, n_iterations) :
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        

    def fit(self, X, Y) :
        # No. of training examples and features
        self.m, self.n = X.shape
        # Initializing weights and bias
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        #implementing gradient descent for optimization 
        for i in range(self.n_iterations) :
            self.update_weights()

    def update_weights(self) :
        #Y_hat formula (sigmoid function)
        Y_hat = 1/(1+np.exp(-(self.X.dot(self.w) + self.b)))

        #derivative of the sigmoid function
        dw = (1/self.m) * np.dot(self.X.T, (Y_hat - self.Y))  # Gradient w.r.t weights
        db = (1/self.m) * np.sum(Y_hat - self.Y)
        
        # Updating weights and bias
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db


    def predict(self) : 
        Y_pred = 1/(1+np.exp(-(self.X.dot(self.w) + self.b)))
        Y_pred = np.where(Y_pred > 0.5, 1, 0)  # Convert probabilities to binary output
        return Y_pred   
    
       


