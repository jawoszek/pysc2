from pysc2.agents.network.generic_network import prepare_network, train_network, run_network

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

inputs = list(map(lambda x: normalize_values(x, mines, maxes), inputs))
outputs = list(map(lambda x: normalize_class(x), outputs))

network = prepare_network(2, [4, 3, 3])
# network.print_weights()

train_network(network, inputs, outputs, 60000)

test_data = normalize_values([5.0,3.5,1.3,0.3], mines, maxes)
output = run_network(network, test_data)

print(choose_class_by_result(output))
