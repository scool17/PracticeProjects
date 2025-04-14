import seaborn as sns
import matplotlib.pyplot as plt
import io
import os
from learnings.neural_networks.tensorflow_healthify import TensorFlow
from flask import send_file
from tensorflow.keras.utils import plot_model

class TensorFlowServer:

    def __init__(self):
        self.tf = TensorFlow()
        self.df = self.tf.df
        self.sequential_model = self.tf.create_sequential_model()
        self.functional_model = self.tf.create_functional_model()
        self.history = self.tf.train_model(epochs=10)

    def correlation_heatmap(self):
        plt.figure(figsize=(12,8))
        plt.title("Correlation Heatmap of Healthify")
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close() 
        return {"image_path": img}

    def plot_sequential_model(self):
        try:
            img = io.BytesIO()
            # image_path = os.path.join("static", "sequential_model.png")
            plot_model(self.sequential_model, to_file=img, show_shapes=True, show_layer_names=True)
            img.seek(0)
            return {"image_path": img}
        except Exception as e:
            return f"Error: {str(e)}"

    def sequential_model_summary(self):
        return self.get_model_summary(self.sequential_model)

    def plot_healthify_loss(self):
        try:
            # Plot loss curves
            plt.figure(figsize=(10, 6))
            plt.plot(self.history.epoch, self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.epoch, self.history.history['val_loss'], label='Validation Loss')
            plt.title("Loss vs Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            
            # Save the figure
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plt.close()
            return {"image_path": img}
        except Exception as e:
            return f"Error: {str(e)}"

    def plot_healthify_accuracy(self):
        try:
            # Plot loss curves
            plt.figure(figsize=(10, 6))
            plt.plot(self.history.epoch, self.history.history['accuracy'], label='Training accuracy')
            plt.plot(self.history.epoch, self.history.history['val_accuracy'], label='Validation accuracy')
            plt.legend()
            plt.title("Accuracy vs Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plt.close()
            return {"image_path": img}
        except Exception as e:
            return f"Error: {str(e)}"

    def plot_functional_model(self):
        try:
            image_path = os.path.join("static", "functional_model.png")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            plot_model(self.functional_model, to_file=image_path, show_shapes=True, show_layer_names=True)
            return {"image_path": image_path}
        except Exception as e:
            return f"Error: {str(e)}"

    def functional_model_summary(self):
        return self.get_model_summary(self.functional_model)
    
    def get_model_summary(self, model):
        # Create a StringIO object to capture the summary output
        stream = io.StringIO()
        # Call model.summary() with a custom print function that writes to the stream
        model.summary(print_fn=lambda x: stream.write(x + "\n"))
        # Get the full summary as a string
        summary_string = stream.getvalue()
        stream.close()
        return summary_string