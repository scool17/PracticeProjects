import seaborn as sns
import matplotlib.pyplot as plt
import io
from learnings.neural_networks.TensorFlow import TensorFlow
from flask import send_file, Flask

def tensorflow():
    tf = TensorFlow()
    return tf.df


def correlation_heatmap():
    df = tensorflow()
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr, annot=True, cmap="coolwarm", fmt=".2f")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()  # Close the figure to free memory

    return send_file(img, mimetype='image/png')