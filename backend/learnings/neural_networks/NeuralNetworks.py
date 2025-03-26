import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetworks:
    def __init__(self):
        self.data = self.read_data()
        

    def formulate_data(self):
        """
        Formulates the data for use in the neural network.

        X is the input features, an m x d matrix where m is the number of samples and d is the number of features.

        Y is the output class, an m x 1 vector where m is the number of samples.

        W is the weights, an d x n matrix where d is the number of features and n is the number of classes(labelled output).

        b is the bias term, an 1 x n vector where n is the number of classes.

        Returns X, Y, W, b
        """
        X = self.data.iloc[:, :-1].to_numpy()
        Y = self.data.iloc[:, -1].to_numpy()

        m, d = X.shape
        n = len(np.unique(Y))

        W = 0.01 * np.random.randn(d, n)
        b = np.zeros((1, n))
        
        return X, Y, W, b

    def explore_data(self):
        pass

    def read_data(self):
        return pd.read_csv('data/spiral.csv')
    
    def calculate_z(self, W, X, b):
        return np.dot(X, W) + b
    
    def softmax(self):
        X, Y, W, b = self.formulate_data()
        z = self.calculate_z(W, X, b)
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    
    def scatterplot(self):

        plt.scatter(self.data["x1"], self.data["x2"], c=self.data["y"], s=40, cmap=plt.cm.Spectral)
        plt.show()