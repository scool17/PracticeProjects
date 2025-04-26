import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.models import Sequential
import pandas as pd
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

class Amazon:

    def __init__(self):
        self.df = self.read_data()

    def read_data(self):
        return pd.read_csv('data/Amazon.csv')
    
    def formulate_data(self):
        X = self.df.drop(columns=['ID', 'Returned'])
        y = self.df['Returned']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def encoding(self):
        encoder = TargetEncoder()
        X_train, X_val, X_test, y_train, y_val, y_test = self.formulate_data()
        X_train = encoder.fit_transform(X_train, y_train)
        X_val = encoder.transform(X_val)
        X_test = encoder.transform(X_test)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scaling(self):
        X_train, X_val, X_test, y_train, y_val, y_test = self.encoding()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train, y_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_model(self, X_train, y_train, X_val, y_val):
        L2reg = tf.keras.regularizers.l2(l2=1e-2)
        model = Sequential()
        model.add(Dense(256, kernel_regularizer=L2reg))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, kernel_regularizer=L2reg))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, kernel_regularizer=L2reg))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_val, y_val), verbose=1, callbacks=[self.early_stop()])
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, accuracy
    
    def run(self):
        X_train, X_val, X_test, y_train, y_val, y_test = self.scaling()
        model = self.create_model(X_train, y_train, X_val, y_val)
        loss, accuracy = self.evaluate_model(model, X_test, y_test)
        return loss, accuracy
    
    def early_stop(self):
        return EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
