import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetworks:
    def __init__(self):
        self.data = self.read_data()
        self.X, self.Y, self.d, self.n = self.formulate_data()
        self.W = 0.01 * np.random.randn(self.d, self.n)
        self.b = np.zeros((1, self.n))
        self.loss = []        

    def read_data(self):
        return pd.read_csv('data/spiral.csv')
    
    def formulate_data(self):
        """
        Formulate the data from the pandas dataframe into X, Y, d, n format.
        
        Parameters
        ----------
        self.data : pandas.DataFrame
            The data to be formulated.
        
        Returns
        -------
        X : numpy.ndarray
            The features of the data.
        Y : numpy.ndarray
            The labels of the data.
        d : int
            The number of features.
        n : int
            The number of classes.
        """
        X = self.data.iloc[:, :-1].to_numpy()
        Y = self.data.iloc[:, -1].to_numpy()
        return X, Y, X.shape[1], len(np.unique(Y))

    def forward_propagation(self):
        z = np.dot(self.X, self.W) + self.b
        exp_z = np.exp(z)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return probs
 
    def backprops(self, probs):
        m = self.Y.shape[0]
        dz = probs
        dz[range(m), self.Y] -= 1
        dz = dz / m

        dW = np.dot(self.X.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        return dW, db
    
    def fit(self, epochs=1000, learning_rate=0.01):
        # print(learning_rate)
        for i in range(epochs):
            probs = self.forward_propagation()
            dW, db = self.backprops(probs)
            self.W = self.W - learning_rate * dW
            self.b = self.b - learning_rate * db
            loss = self.calculate_loss(probs)
            self.loss.append(loss)
        self.history = pd.DataFrame({
        'step': list(range(epochs)),
        'loss': self.loss})

    def predict(self):
        probs = self.forward_propagation()
        return np.argmax(probs, axis=1)

    def calculate_loss(self, probs):
        m = self.Y.shape[0]
        error = -np.log(probs[range(m), self.Y])
        return np.sum(error)/m
    
    def explore_data(self):
        pass
    
    def scatterplot(self):
        plt.scatter(self.data["x1"], self.data["x2"], c=self.data["y"], s=40, cmap=plt.cm.Spectral)
        plt.show()