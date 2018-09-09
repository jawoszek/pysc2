from ast import literal_eval
from pysc2.agents.network.custom_network import train_network, run_network, Source, Neuron, NeuronLayer, NeuralNetwork
import random
from pysc2.agents.network.terran_bot_parser import BotDataParser, BUILD_NODES, RECRUIT_NODES

TOTAL_NUMBER_OF_BASE_NEURONS = BUILD_NODES * 2 + RECRUIT_NODES


parser = BotDataParser()
inputs, outputs = parser.read_data_file('results_random_very_easy.txt')
t_inputs, t_outputs = parser.read_data_file('tests_random_very_easy.txt')


def prepare_network():
    random.seed()
    layers = []
    neurons_in_base = list(map(lambda _: Source(), range(0, TOTAL_NUMBER_OF_BASE_NEURONS)))
    layers.append(NeuronLayer(neurons_in_base))
    middle_layer = []
    for i in range(0, BUILD_NODES):
        mid_neuron = Neuron()
        mid_neuron.add_input(neurons_in_base[i*2])
        mid_neuron.add_input(neurons_in_base[i*2+1])
        neurons_in_base[i*2].add_output(mid_neuron)
        neurons_in_base[i*2+1].add_output(mid_neuron)
        middle_layer.append(mid_neuron)
    layers.append(NeuronLayer(middle_layer))
    last_neuron = Neuron()
    for i in range(BUILD_NODES*2, BUILD_NODES*2+RECRUIT_NODES):
        last_neuron.add_input(neurons_in_base[i])
    for i in middle_layer:
        last_neuron.add_input(i)
    layers.append(NeuronLayer([last_neuron]))
    return NeuralNetwork(layers)


# network = prepare_network(2, [int(BUILD_NODES*2 + RECRUIT_NODES), int((BUILD_NODES*2 + RECRUIT_NODES)/2), 1])
network = prepare_network()
train_network(network, inputs, outputs, 1000)

output = run_network(network, t_inputs)
print("result: {0}    vs    real: {1}".format(output, t_outputs))



