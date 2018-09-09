from ast import literal_eval
from pysc2.agents.network.custom_network import train_network, run_network, Source, Neuron, NeuronLayer, NeuralNetwork
import random

BUILD_NODES = 20
RECRUIT_NODES = 100

UNITS_TO_VALUE = {
    45: 0,
    48: 1
}
BUILDING_TO_VALUE = {
    19: 0,
    20: 0.5,
    21: 1
}


class BotDataParser:

    def normalize_row(self, row, min_pop, max_pop):
        output_norm = []
        for i in range(0, BUILD_NODES):
            pop = row[i*2]
            building = row[i*2+1]
            output_norm.append((pop-min_pop)/(max_pop-min_pop))
            output_norm.append(BUILDING_TO_VALUE[building])
        for i in range(BUILD_NODES*2,BUILD_NODES*2+RECRUIT_NODES):
            unit = row[i]
            output_norm.append(UNITS_TO_VALUE[unit])
        return output_norm

    def normalize_output_row(self, row):
        return [1 if i == 1 else 0 for i in row]

    def read_data_file(self, path):
        min_pop = 10
        max_pop = 100
        inputs_read = []
        outputs_read = []
        with open(path, 'r') as file:
            for line in file.readlines():
                whole_line = "({})".format(line)
                whole_object = literal_eval(whole_line)
                whole_record = []
                for pop, building in whole_object[0]:
                    whole_record.append(pop)
                    whole_record.append(building)
                    # if pop < min_pop:
                    #     min_pop = pop
                    # if pop > max_pop:
                    #     max_pop = pop
                for unit in whole_object[1]:
                    whole_record.append(unit)
                inputs_read.append(whole_record)
                outputs_read.append([whole_object[2]])

        inputs_read = list(map(lambda row: self.normalize_row(row, min_pop, max_pop), inputs_read))
        outputs_read = list(map(self.normalize_output_row, outputs_read))
        return inputs_read, outputs_read

    def read_input_from_text(self, text):
        min_pop = 10
        max_pop = 100
        inputs_read = []
        for line in text.split('\n'):
            whole_line = "({})".format(line)
            whole_object = literal_eval(whole_line)
            whole_record = []
            for pop, building in whole_object[0]:
                whole_record.append(pop)
                whole_record.append(building)
                # if pop < min_pop:
                #     min_pop = pop
                # if pop > max_pop:
                #     max_pop = pop
            for unit in whole_object[1]:
                whole_record.append(unit)
            inputs_read.append(whole_record)

        inputs_read = list(map(lambda row: self.normalize_row(row, min_pop, max_pop), inputs_read))
        return inputs_read
