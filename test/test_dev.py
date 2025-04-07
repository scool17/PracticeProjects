


from learnings.neural_networks.neural_networks import NeuralNetworks
# import configparser
from learnings.neural_networks.tensorflow import TensorFlow
from learnings.neural_networks.mnist_dataset import HandWrittenDigits

from learnings.neural_networks.amazon import Amazon

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

    # hwd = HandWrittenDigits()

    # model = hwd.train_model(12)

    # predictions = hwd.predict_result(model)
    # print(hwd.check_accuracy(predictions))

    am = Amazon()
    print(am.run())

