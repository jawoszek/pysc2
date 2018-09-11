from pysc2.agents.network.custom_network import prepare_dense_network, train_network, run_network
import random
import math
from numpy import array
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

classes = {'Iris-setosa': [1.0, 0.0, 0.0], 'Iris-versicolor': [0.0, 1.0, 0.0], 'Iris-virginica': [0.0, 0.0, 1.0]}
random.seed()

class SOM:

    def __init__(self, size, inputsCount, max_iter) -> None:
        super().__init__()
        self.start_learn = 0.1
        self.current_iter = 0
        self.max_iter = max_iter
        self.size = size
        self.nodes = [[SOMNode(inputsCount) for _ in range(0, size)] for _ in range(0, size)]

    def process(self, input_vector):
        self.current_iter += 1
        # print(self.get_range())
        c_x, c_y = self.select_unit(input_vector)
        c_range = self.get_range()
        for x in range(0, self.size):
            for y in range(0, self.size):
                unit_dist = self.unit_distance_from(x, y, c_x, c_y)
                if not unit_dist < c_range:
                    continue
                self.apply_node(x, y, c_x, c_y, input_vector)

    def select_unit(self, input_vector):
        c_x, c_y = 0, 0
        current_distance = self.nodes[c_x][c_y].distance(input_vector)
        for x in range(0, self.size):
            for y in range(0, self.size):
                if self.nodes[x][y].distance(input_vector) < current_distance:
                    c_x = x
                    c_y = y
                    current_distance = self.nodes[x][y].distance(input_vector)
        return c_x, c_y

    def unit_distance_from(self, x, y, u_x, u_y):
        return math.sqrt((x - u_x)**2 + (y - u_y)**2)

    def get_range(self):
        return self.size*math.exp(-self.current_iter/(self.max_iter/math.log(self.size)))

    def apply_node(self, x, y, c_x, c_y, input_vector):
        node = self.nodes[x][y]
        c_dist = self.unit_distance_from(x, y, c_x, c_y)
        dist_val = math.exp(-(c_dist**2)/(2*(self.get_range()**2)*self.current_iter))
        learn = self.start_learn * math.exp(-self.current_iter/(self.max_iter/math.log(self.size)))
        # print(learn)
        node.weights = node.weights + dist_val * learn * (input_vector - node.weights)

    def print_grid(self):
        [[print(node.weights) for node in nodes_x] for nodes_x in self.nodes]
        max_c = max([max([max(node.weights) for node in nodes_x]) for nodes_x in self.nodes])
        min_c = min([min([min(node.weights) for node in nodes_x]) for nodes_x in self.nodes])

        data = [[[(value - min_c)/(max_c - min_c) for value in node.weights[1:4]] for node in nodes_x] for nodes_x in self.nodes]

        plt.imshow(data)
        plt.show()

class SOMNode:

    def __init__(self, weightsCount) -> None:
        super().__init__()
        self.size = weightsCount
        self.weights = array([random.uniform(0, 1) for i in range(0, weightsCount)])

    def distance(self, input_vector):
        return math.sqrt(sum([(input_vector[i] - self.weights[i])**2 for i in range(0, self.size)]))


def choose_class_by_result(result):
    keys_with_sum = dict(map(lambda pair: (pair[0], get_sums_of_differences(result, pair[1])), classes.items()))
    return min(keys_with_sum, key=lambda k: keys_with_sum[k])


def get_sums_of_differences(result, value):
    return sum(different_values(result, value))


def different_values(result, value):
    return list(map(lambda x: abs(x[0]-x[1]), zip(result, value)))


def normalize_value(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)


def normalize_values(values, mines, maxes):
    output = []
    for i in range(0, len(values)):
        value = values[i]
        min_value = mines[i]
        max_value = maxes[i]
        output.append(normalize_value(value, min_value, max_value))
    return output


def normalize_class(name):
    return classes[name]


inputs = []
outputs = []
with open('iris.data', 'r') as file:
    for line in file.readlines():
        fields = line.split(',')
        inputs_from_line = list(map(lambda x: float(x), fields[:-1]))
        outputs_from_line = fields[-1].replace('\n', '').replace('\r', '')

        inputs.append(inputs_from_line)
        outputs.append(outputs_from_line)

maxes = [0] * 4
mines = [1000] * 4
for line in inputs:
    for i in range(0, len(line)):
        value = line[i]
        if value < mines[i]:
            mines[i] = value
        if value > maxes[i]:
            maxes[i] = value

inputs = list(map(lambda x: normalize_values(x, mines, maxes), inputs))
outputs = list(map(lambda x: normalize_class(x), outputs))

som = SOM(50, 4, 1000)
for i in range(0, 1000):
    som.process(array(random.choice(inputs)))

som.print_grid()

print(inputs)
print(outputs)


# network = prepare_dense_network([4, 3, 3])
# # print_weights(network)
#
# train_network(network, inputs, outputs, 10000)
# # custom_network.print_weights(network)
#
# test_data = normalize_values([5.8,2.7,4.1,1.0], mines, maxes)
# output = run_network(network, test_data)
#
# print(output)
# print(choose_class_by_result(output))
