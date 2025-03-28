


from backend.netflix.Netflix import Netflix
from backend.utility.get_data import Parser
from backend.learnings.neural_networks.NeuralNetworks import NeuralNetworks
# import configparser

if __name__ == "__main__":

    # p = Parser()
    

    # netflix = Netflix()

    # print(netflix.netflix_data())

    nn = NeuralNetworks()
    nn.fit(learning_rate=1)
    print(nn.loss)
