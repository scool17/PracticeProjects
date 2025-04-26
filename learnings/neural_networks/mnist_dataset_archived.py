from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler

class MnistData:
    def __init__(self, dataset):
        (X_train, y_train), (X_test, y_test) = dataset()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        tf.random.set_seed(42)

    def create_model(self):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_uniform'),
            keras.layers.Dense(64, activation='relu', kernel_initializer='glorot_uniform'),
            keras.layers.Dense(32, activation='relu', kernel_initializer='glorot_uniform'),
            keras.layers.Dense(16, activation='relu', kernel_initializer='glorot_uniform'),
            keras.layers.Dense(10, activation='softmax')
        ])

        # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer=tf.keras.optimizers.Adam(beta_1= 0.9, beta_2= 0.999), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model
    
    
    def train_model(self, epochs=10):
        model = self.create_model()
        lrdecay = LearningRateScheduler(scheduler)
        history = model.fit(self.X_train, self.y_train, epochs=epochs, verbose=1, validation_split=0.1, batch_size=256, callbacks=[lrdecay])
        return model
    
    def predict_result(self, model):
        pred = model.predict(self.X_test)
        predictions= [np.argmax(i) for i in pred]
        return predictions
    
    def check_accuracy(self, predictions):
        accuracy = np.sum(predictions == self.y_test) / len(self.y_test)
        return accuracy

class HandWrittenDigits(MnistData):
    
    def __init__(self):
        super().__init__(keras.datasets.mnist.load_data)
    

def scheduler(epoch, lr):
    ro = 0.01
    lr = (1 / (1 + ro * epoch)) * lr
    return lr 