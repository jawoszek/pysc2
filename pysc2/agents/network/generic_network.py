from numpy import exp, array, random, dot


class NeuronLayer:
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.layers_count = len(layers)

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):

            outputs = self.think(training_set_inputs)
            layers_errors = [0] * self.layers_count
            layers_delta = [0] * self.layers_count
            layers_adjustment = [0] * self.layers_count
            for i in range(self.layers_count-1, -1, -1):
                if i == self.layers_count-1:
                    layers_errors[i] = training_set_outputs - outputs[i]
                else:
                    layers_errors[i] = layers_delta[i+1].dot(self.layers[i+1].synaptic_weights.T)
                layers_delta[i] = layers_errors[i] * self.__sigmoid_derivative(outputs[i])
                
            for i in range(0, self.layers_count):
                if i == 0:
                    layers_adjustment[i] = training_set_inputs.T.dot(layers_delta[i])
                else:
                    layers_adjustment[i] = outputs[i-1].T.dot(layers_delta[i])

            for i in range(0, self.layers_count):
                self.layers[i].synaptic_weights += layers_adjustment[i]

    def think(self, inputs):
        input_to_layer = inputs
        outputs = []
        for i in range(0, self.layers_count):
            output = self.__sigmoid(dot(input_to_layer, self.layers[i].synaptic_weights))
            outputs.append(output)
            input_to_layer = output
        return outputs

    def print_weights(self):
        for i in range(0, self.layers_count):
            print('layer {0} weights:'.format(i))
            print(self.layers[i].synaptic_weights)


def prepare_network(layers_count, neurons_count):
    random.seed()
    layers = []
    for i in range(0, layers_count):
        layers.append(NeuronLayer(neurons_count[i+1], neurons_count[i]))

    return NeuralNetwork(layers)


def train_network(network, inputs, outputs, iteration_count):
    training_set_inputs = array(inputs)
    training_set_outputs = array(outputs)

    network.train(training_set_inputs, training_set_outputs, iteration_count)


def run_network(network, inputs):
    return network.think(inputs)[-1]
