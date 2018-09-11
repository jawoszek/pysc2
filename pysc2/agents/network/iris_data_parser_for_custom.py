from pysc2.agents.network.custom_network import prepare_dense_network, train_network, run_network

classes = {'Iris-setosa': [1.0, 0.0, 0.0], 'Iris-versicolor': [0.0, 1.0, 0.0], 'Iris-virginica': [0.0, 0.0, 1.0]}


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

outputs_names = outputs
inputs = list(map(lambda x: normalize_values(x, mines, maxes), inputs))
outputs = list(map(lambda x: normalize_class(x), outputs))


network = prepare_dense_network([4, 6, 5, 3])
# print_weights(network)

train_network(network, inputs, outputs, 500)
# custom_network.print_weights(network)

test_data = normalize_values([5.8,2.7,4.1,1.0], mines, maxes)
output = run_network(network, test_data)
print(output)

# received_names = []
# count = 0
# for i in range(0, len(outputs_names)):
#     output = run_network(network, normalize_values(inputs[i], mines, maxes))
#     received_names.append(choose_class_by_result(output))
#
# print(count)
# print(received_names)
# print(outputs_names)
# test_data = list(map(lambda input: normalize_values(input, mines, maxes), inputs))
# output = run_network(network, test_data)
#
# print(len(outputs_names))
# print(len(output))
#
# print(outputs_names)
# received_names = list(map(lambda e: choose_class_by_result(e), output))
# print(received_names)

# hits = sum([1 for i in range(0, len(received_names)) if outputs_names[i] == received_names[i]])
# print(hits)
