import pandas as pd
import sklearn
import tensorflow as tf
# import keras
from keras import layers
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os



class TensorFlow:

    def __init__(self):
        self.data = self.read_data()
        self.df = self.data.copy()
        self.formulate_data()

    def read_data(self):
        return pd.read_csv('data/healthyfime_tfkeras.csv')
        
    def formulate_data(self):
        self.df.replace({"M":0, "F":1} , inplace = True)
        classes = list(self.df['class'].unique())
        mapping_dict = { ch : i for i, ch in enumerate(sorted(classes, reverse=True)) }
        self.df['class'].replace(mapping_dict , inplace = True)
        return self.df
    
    def create_model(self):
        model = tf.keras.Sequential([
            layers.Dense(16, activation='relu', input_shape=(1,)),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(loss=tf.keras.losses.mse, optimizer=tf.keras.optimizers.Adam(0.01))
        return model