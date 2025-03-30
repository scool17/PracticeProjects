import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.utils import plot_model

class TensorFlow:

    def __init__(self):
        self.data = self.read_data()
        self.df = self.data.copy()
        self.formulate_data()
        self.X_train, self.X_test, self.y_train,  self.y_test = self.split_data()

    def read_data(self):
        return pd.read_csv('data/healthyfime_tfkeras.csv')

    def formulate_data(self):
        self.df.replace({"M":0, "F":1} , inplace = True)
        classes = list(self.df['class'].unique())
        mapping_dict = { ch : i for i, ch in enumerate(sorted(classes, reverse=True)) }
        self.df['class'].replace(mapping_dict , inplace = True)
        return self.df
    
    def split_data(self):
        X = self.df.iloc[:, :-1].to_numpy()
        y = self.df.iloc[:, -1].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train, y_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test
    
    def create_model(self):
        model = Sequential([Dense(64, activation='relu', input_shape=(11,), name="Hidden_Layer_1"),
                            Dense(4, activation='softmax', name="Output_Layer"),
        ])
        model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(0.01), metrics=["accuracy"])
        return model
    
    def train_model(self):
        model = self.create_model()
        history = model.fit(self.X_train, self.y_train, epochs= 100, validation_split=0.1, verbose=1, batch_size=256)
        return history