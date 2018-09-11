from keras.models import Sequential
from keras.layers import Dense, Dropout
from numpy import array
from pysc2.agents.network.terran_bot_parser import BotDataParser, BUILD_NODES, RECRUIT_NODES
from keras.utils import to_categorical
import random

TOTAL_NUMBER_OF_BASE_NEURONS = BUILD_NODES * 2 + RECRUIT_NODES


class SimpleKerasTest:

    def learn_and_cross_validate(self, path, epochs=100):
        model = Sequential()
        model.add(Dense(units=64, activation='relu', input_dim=140))
        # self.model.add(Dropout(0.5))
        # self.model.add(Dense(64, activation='relu'))
        # self.model.add(Dropout(0.5))
        model.add(Dense(units=2, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])

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
            x_train = array(train_input)
            y_train = to_categorical(array(train_output).T[0])
            model.fit(x_train, y_train, epochs=epochs, batch_size=256)
            x_test = array(test_input)
            results = model.predict(x_test, batch_size=256)
            results_single = self.check_results(results, test_output)
            results_all.append(results_single)
        results_sum = 0
        for classes in results_all:
            percent = len([c for c in classes if c[0]]) / len(classes) * 100
            print("{0}% success".format(percent))
            results_sum += percent
        print("average success: {0}%".format(results_sum / k))
        with open('cross_validation_dense.txt', "a") as file:
            file.write("{0},{1}\n".format(epochs, results_sum / k))

    def check_results(self, results, output_t):
        return [(self.is_class_hit(results[i], output_t[i][0]), output_t[i][0], results[i]) for i in range(0, len(results))]

    def is_class_hit(self, result, real_result):
        resulting_class = 1 if result[1] > result[0] else 0
        return resulting_class == real_result


if __name__ == "__main__":
    network = SimpleKerasTest()
    for i in range(1, 400):
        network.learn_and_cross_validate('../results_random_very_easy.txt', i)
