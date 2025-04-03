import seaborn as sns
import matplotlib.pyplot as plt
import io
from learnings.neural_networks.TensorFlow import TensorFlow
from flask import send_file
from tensorflow.keras.utils import plot_model

class TensorFlowServer:

    def __init__(self):
        self.tf = TensorFlow()
        self.df = self.tf.df
        self.sequential_model = self.tf.create_sequential_model()
        self.history = self.tf.train_model(epochs=10)

    def correlation_heatmap(self):
        plt.figure(figsize=(12,8))
        plt.title("Correlation Heatmap of Healthify")
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()  # Close the figure to free memory
        return {"image_path": img}

    def plot_sequential_model(self):
        plot_model(self.tf.create_sequential_model(), to_file='model.png', show_shapes=True, show_layer_names=True)
        return {"image_path": 'model.png'}
    
    def sequential_model_summary(self):
        return self.get_model_summary(self.sequential_model)

    def plot_healthify_loss(self):
        plt.plot(self.history.epoch, self.history.history['loss'], label='training loss')
        plt.plot(self.history.epoch, self.history.history['val_loss'], label='validation loss')
        plt.legend()
        plt.title("Loss vs Epochs")
        plt.show()
        return send_file('model.png', mimetype='image/png')

    def plot_healthify_accuracy(self):
        plt.plot(self.history.epoch, self.history.history['accuracy'], label='training accuracy')
        plt.plot(self.history.epoch, self.history.history['val_accuracy'], label='validation accuracy')
        plt.legend()
        plt.title("Accuracy vs Epochs")
        plt.show()
        return send_file('model.png', mimetype='image/png')

    def plot_functional_model(self):
        plot_model(self.tf.create_functional_model(), to_file='model.png', show_shapes=True, show_layer_names=True)
        return send_file('model.png', mimetype='image/png')

    def functional_model_summary(self):
        return self.tf.create_functional_model().summary()
    
    def get_model_summary(self, model):
        # Create a StringIO object to capture the summary output
        stream = io.StringIO()
        # Call model.summary() with a custom print function that writes to the stream
        model.summary(print_fn=lambda x: stream.write(x + "\n"))
        # Get the full summary as a string
        summary_string = stream.getvalue()
        stream.close()
        return summary_string