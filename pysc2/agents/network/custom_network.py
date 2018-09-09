from numpy import exp, array, random, clip

random.seed()


def sigmoid(x):
    # print x
    # return array(list(map(lambda x: 1 / (1 + exp(-x)), x)))
    # x = clip(x, -500, 500)
    return 1 / (1 + exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class Source:
    def __init__(self, output=0):
        self.output = output
        self.outputs = list()

    def set_output(self, output):
        self.output = output

    def get_output(self):
        return self.output
        
    def add_output(self, output):
        self.outputs.append(output)


class Neuron(Source):
    def __init__(self):
        Source.__init__(self)
        self.inputs = list()
        
    def get_weight_for_input(self, input):
        for pair in self.inputs:
            if pair[0] == input:
                return pair[1]

    def add_input(self, input):
        weight = 2*random.random()-1
        # print weight
        self.inputs.append((input, weight))

    def calculate_output(self):
        self.output = sigmoid(array(list(map(lambda x: array(x[0].get_output()) * x[1], self.inputs))).sum(axis=0))

    def calculate_error(self):
        self.error = array(list(map(lambda x: x.get_delta() * x.get_weight_for_input(self), self.outputs))).sum(axis=0)
        self.diff = self.error * sigmoid_derivative(self.output)
        
    def calculate_error_first(self, training_output):
        self.error = training_output - self.output
        self.diff = self.error * sigmoid_derivative(self.output)
        
    def get_delta(self):
        return self.diff

    def calculate_adjustment(self):
        self.inputs = list(map(lambda input: self.get_adjustment(input), self.inputs))
        # self.adjustment = list(map(lambda input: input[0].get_output(), self.inputs)) * self.get_delta()

    def get_adjustment(self, input):
        return input[0], input[1] + sum([x * y for x, y in zip(input[0].get_output(), self.get_delta())])


class NeuronLayer:
    def __init__(self, neurons):
        self.neurons = neurons

    def calculate_outputs(self):
        for neuron in self.neurons:
            neuron.calculate_output()
            
    def calculate_errors(self):
        for neuron in self.neurons:
            neuron.calculate_error()
        
    def calculate_errors_first(self, training_set_output):
        for i in range(0, len(self.neurons)):
            self.neurons[i].calculate_error_first(training_set_output[i])
        
    def set_outputs(self, outputs):
        for i in range(0, len(self.neurons)):
            self.neurons[i].set_output(outputs[i])

    def calculate_adjustment(self):
        for neuron in self.neurons:
            neuron.calculate_adjustment()
            
    def get_outputs(self):
        return array(list(map(lambda neuron: neuron.get_output(), self.neurons))).T


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.layers_count = len(layers)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
        
            self.think(training_set_inputs)

            self.layers[-1].calculate_errors_first(training_set_outputs)
            for i in range(self.layers_count-2, 0, -1):
                self.layers[i].calculate_errors()

            for i in range(1, self.layers_count):
                self.layers[i].calculate_adjustment()

    def think(self, inputs):
        self.layers[0].set_outputs(inputs)
        for layer in self.layers[1:]:
                layer.calculate_outputs()
        return self.layers[-1].get_outputs()
        
    def get_layers(self):
        return self.layers[1:]

        
def print_weights(network):
    weights = list(map(lambda inputs: map(lambda input: input[1], inputs), map(lambda neurons: map(lambda neuron: neuron.inputs, neurons), map(lambda layer: layer.neurons, network.get_layers()))))
    
    print('weights:')
    print(weights)


def prepare_dense_network(neurons_count):
    random.seed()
    layers = []
    neurons_in_layer = list(map(lambda _: Source(), range(0, neurons_count[0])))
    previous = neurons_in_layer
    layers.append(NeuronLayer(neurons_in_layer))
    for i in neurons_count[1:]:
        neurons_in_layer = list(map(lambda _: Neuron(), range(0, i)))
        for neuron in neurons_in_layer:
            for previous_neuron in previous:
                neuron.add_input(previous_neuron)
                previous_neuron.add_output(neuron)
        previous = neurons_in_layer
        layers.append(NeuronLayer(neurons_in_layer))
    return NeuralNetwork(layers)


def train_network(network, inputs, outputs, iteration_count):
    training_set_inputs = array(inputs)
    training_set_outputs = array(outputs)

    network.train(training_set_inputs.T, training_set_outputs.T, iteration_count)


def run_network(network, inputs):
    return network.think(array(inputs).T)
