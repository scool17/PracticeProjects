from flask import Flask, request, jsonify, send_file
from waitress import serve

from learnings.neural_networks.NeuralNetworks import NeuralNetworks
from flask_server.tensorflow_server import TensorFlowServer

app = Flask(__name__)

tf_server = TensorFlowServer()

nn = NeuralNetworks()
nn.fit(learning_rate=1)

@app.route('/')
def home():
    return "Welcome to my Project new"

@app.route('/predict')
def predict():
    return [nn.loss[-1]]

@app.route('/show')
def show():
    return jsonify(nn.data.to_dict(orient='records'))

@app.route('/tensorflow/healthify_heatmap')
def healthify_heatmap():
    response =  tf_server.correlation_heatmap()
    if isinstance(response, str) and response.startswith("Error"):
        return jsonify({"error": response}), 400
    return send_file(response["image_path"], mimetype='image/png')

@app.route('/tensorflow/healthify_sequential_model')
def healthify_sequential_model():
    response =  tf_server.plot_sequential_model()
    if isinstance(response, str) and response.startswith("Error"):
        return jsonify({"error": response}), 400
    return send_file(response["image_path"], mimetype='image/png')

@app.route('/tensorflow/healthify_sequential_model_summary')
def healthify_sequential_model_summary():
    summary = tf_server.sequential_model_summary()
    return "<pre>{}</pre>".format(summary)

@app.route('/tensorflow/healthify_loss')
def healthify_loss():
    return send_file(tf_server.plot_healthify_loss(), mimetype='image/png')

@app.route('/tensorflow/healthify_accuracy')
def healthify_accuracy():    
    return send_file(tf_server.plot_healthify_accuracy(), mimetype='image/png')

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=8000)