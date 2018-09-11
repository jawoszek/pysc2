from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, concatenate
from numpy import array
from pysc2.agents.network.terran_bot_parser import BotDataParser, BUILD_NODES, RECRUIT_NODES
from keras.utils import to_categorical
import random

TOTAL_NUMBER_OF_BASE_NEURONS = BUILD_NODES * 2 + RECRUIT_NODES


def construct_network():
    buildings_1_input = Input(shape=(10,), dtype='float32', name='buildings_1')  # first 5 buildings
    buildings_2_input = Input(shape=(10,), dtype='float32', name='buildings_2')  # second 5 buildings
    buildings_3_input = Input(shape=(10,), dtype='float32', name='buildings_3')  # third 5 buildings
    buildings_4_input = Input(shape=(10,), dtype='float32', name='buildings_4')  # fourth 5 buildings
    units_input = Input(shape=(100,), dtype='float32', name='units')  # units
    inputs = [buildings_1_input, buildings_2_input, buildings_3_input, buildings_4_input, units_input]
    b_1_l = Dense(64, activation='relu')(buildings_1_input)
    b_2_l = Dense(64, activation='relu')(buildings_2_input)
    b_3_l = Dense(64, activation='relu')(buildings_3_input)
    b_4_l = Dense(64, activation='relu')(buildings_4_input)
    u_l = Dense(64, activation='relu')(units_input)
    merged = concatenate([b_1_l, b_2_l, b_3_l, b_4_l, u_l])
    main_l = Dense(64, activation='relu')(merged)
    main_output = Dense(2, activation='softmax', name='main_output')(main_l)
    model = Model(inputs=inputs, outputs=[main_output])
    # model.compile(optimizer='rmsprop', loss='binary_crossentropy',
    #               loss_weights=[1., 0.2])
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model


def split_input(input):
    inputs_1 = array(list(map(lambda x: x[0:10], input)))
    inputs_2 = array(list(map(lambda x: x[10:20], input)))
    inputs_3 = array(list(map(lambda x: x[20:30], input)))
    inputs_4 = array(list(map(lambda x: x[30:40], input)))
    inputs_5 = array(list(map(lambda x: x[40:140], input)))
    return [inputs_1, inputs_2, inputs_3, inputs_4, inputs_5]


class SimpleKerasTest:

    def learn_and_cross_validate(self, path, epochs=100):
        model = construct_network()

        parser = BotDataParser()
        input, output = parser.read_data_file(path)

        indexes = list(range(0, len(input)))
        random.shuffle(indexes)
        k = 5
        segment_length = int(len(input)/k)
        results_all = []
        for i in range(0, k):
            test_range = range(i*segment_length, (i+1)*segment_length)
            indexes_l = [indexes[index] for index in range(0, len(indexes)) if index not in test_range]
            indexes_t = [indexes[index] for index in range(0, len(indexes)) if index in test_range]
            train_input = [input[index] for index in range(0, len(input)) if index in indexes_l]
            train_output = [output[index] for index in range(0, len(input)) if index in indexes_l]
            test_input = [input[index] for index in range(0, len(input)) if index in indexes_t]
            test_output = [output[index] for index in range(0, len(input)) if index in indexes_t]
            x_train = split_input(train_input)
            y_train = to_categorical(array(train_output).T[0])
            model.fit(x_train, y_train, epochs=epochs, batch_size=256)
            x_test = split_input(test_input)
            results = model.predict(x_test, batch_size=256)
            results_single = self.check_results(results, test_output)
            results_all.append(results_single)
        results_sum = 0
        for classes in results_all:
            percent = len([c for c in classes if c[0]]) / len(classes) * 100
            print("{0}% success".format(percent))
            results_sum += percent
        print("average success: {0}%".format(results_sum / k))
        with open('balanced_cross_validation_sparse.txt', "a") as file:
            file.write("{0},{1}\n".format(epochs, results_sum / k))

    def check_results(self, results, output_t):
        return [(self.is_class_hit(results[i], output_t[i][0]), output_t[i][0], results[i]) for i in range(0, len(results))]

    def is_class_hit(self, result, real_result):
        resulting_class = 1 if result[1] > result[0] else 0
        return resulting_class == real_result


if __name__ == "__main__":
    network = SimpleKerasTest()
    for i in range(1, 400):
        network.learn_and_cross_validate('../balanced_results_random_very_easy.txt', i)
