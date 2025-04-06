


from learnings.neural_networks.NeuralNetworks import NeuralNetworks
# import configparser
from learnings.neural_networks.TensorFlow import TensorFlow
from learnings.neural_networks.MnistData import HandWrittenDigits

if __name__ == "__main__":

    # p = Parser()
    

    # netflix = Netflix()

    # print(netflix.netflix_data())

    # nn = NeuralNetworks()
    # nn.fit(learning_rate=1)
    # print(nn.loss)

    # tf = TensorFlow()
    # # history = tf.train_model()
    # model = tf.create_functional_model()
    # model.summary()
    # print(model.weights)
    # print(model.summary())

    hwd = HandWrittenDigits()

    model = hwd.train_model(12)

    predictions = hwd.predict_result(model)
    print(hwd.check_accuracy(predictions))

