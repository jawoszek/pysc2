from pysc2.agents.network.custom_network import prepare_dense_network, train_network, run_network

network = prepare_dense_network([3, 3, 1])

inputs = [[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1]]
outputs = [[1], [1], [0], [0], [1]]

train_network(network, inputs, outputs, 10000)

output = run_network(network, [0, 0, 1])

print(output)
