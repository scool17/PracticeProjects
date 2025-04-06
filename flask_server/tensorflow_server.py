import seaborn as sns
import matplotlib.pyplot as plt
import io
import os
from learnings.neural_networks.TensorFlowHealthify import TensorFlowHelathify
from flask import send_file
from tensorflow.keras.utils import plot_model
import uuid
import os
import glob

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)  

class TensorFlowServer:

    def __init__(self):
        self.tf = TensorFlowHelathify()
        self.df = self.tf.df
        self.sequential_model = self.tf.create_sequential_model()
        self.functional_model = self.tf.create_functional_model()
        self.history = self.tf.train_model(epochs=10)

    def correlation_heatmap(self):
        delete_old_images("correlation_heatmap")
        filename = f"correlation_heatmap_{uuid.uuid4().hex}.png"
        filepath = os.path.join(static_dir, filename)
        plt.figure(figsize=(12,8))
        plt.title("Correlation Heatmap of Healthify")
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.savefig(filepath)
        plt.close() 
        return {"image_path": filepath}

    def plot_sequential_model(self):
        try:
            img = io.BytesIO()
            plot_model(self.sequential_model, to_file=img, show_shapes=True, show_layer_names=True)
            img.seek(0)
            return {"image_path": img}
        except Exception as e:
            return f"Error: {str(e)}"

    def sequential_model_summary(self):
        return self.get_model_summary(self.sequential_model)

    def plot_healthify_loss(self):
        try:
            delete_old_images("healthify_loss")
            filename = f"healthify_loss_{uuid.uuid4().hex}.png"
            filepath = os.path.join(static_dir, filename)
            # Plot loss curves
            plt.figure(figsize=(10, 6))
            plt.plot(self.history.epoch, self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.epoch, self.history.history['val_loss'], label='Validation Loss')
            plt.title("Loss vs Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            # Save the figure
            plt.savefig(filepath)
            plt.close()
            return {"image_path": filepath}
        except Exception as e:
            return f"Error: {str(e)}"

    def plot_healthify_accuracy(self):
        try:
            # Plot loss curves
            delete_old_images("healthify_accuracy")
            filename = f"healthify_accuracy_{uuid.uuid4().hex}.png"
            filepath = os.path.join(static_dir, filename)
            plt.figure(figsize=(10, 6))
            plt.plot(self.history.epoch, self.history.history['accuracy'], label='Training accuracy')
            plt.plot(self.history.epoch, self.history.history['val_accuracy'], label='Validation accuracy')
            plt.legend()
            plt.title("Accuracy vs Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.savefig(filepath)
            plt.close()
            return {"image_path": filepath}
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
    

def delete_old_images(prefix):
    pattern = os.path.join(static_dir, f"{prefix}_*.png")
    for file_path in glob.glob(pattern):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")