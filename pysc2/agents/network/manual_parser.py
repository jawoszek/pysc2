from pysc2.agents.network.generic_network import prepare_network, train_network, run_network

network = prepare_network(2, [3, 3, 1])
network.print_weights()

inputs = [[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 0, 1]]
outputs = [[1], [1], [0], [0]]

train_network(network, inputs, outputs, 1)

output = run_network(network, [0, 1, 1])

print(output)
