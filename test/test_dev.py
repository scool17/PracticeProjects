


from learnings.neural_networks.NeuralNetworks import NeuralNetworks
# import configparser
from learnings.neural_networks.TensorFlowHealthify import TensorFlowHelathify

if __name__ == "__main__":

    # p = Parser()
    

    # netflix = Netflix()

    # print(netflix.netflix_data())

    # nn = NeuralNetworks()
    # nn.fit(learning_rate=1)
    # print(nn.loss)

    tf = TensorFlowHelathify()
    # history = tf.train_model()
    model = tf.create_functional_model()
    model.summary()
    # print(model.weights)
    # print(model.summary())
