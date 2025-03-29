from flask import Flask, request, jsonify
from waitress import serve

from backend.learnings.neural_networks.NeuralNetworks import NeuralNetworks

app = Flask(__name__)

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


if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=8000)