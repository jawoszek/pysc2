from keras.models import Sequential
from keras.layers import Dense, Dropout
from numpy import array
from pysc2.agents.network.terran_bot_parser import BotDataParser, BUILD_NODES, RECRUIT_NODES
from keras.utils import to_categorical
import random

TOTAL_NUMBER_OF_BASE_NEURONS = BUILD_NODES * 2 + RECRUIT_NODES


class SimpleKerasTest:

    def learn_and_test_on_the_same_data(self, path, epochs=100):
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
        x_train = array(input)
        y_train = to_categorical(array(output).T[0])
        model.fit(x_train, y_train, epochs=epochs, batch_size=256)
        results = model.predict(x_train, batch_size=256)
        classes = self.check_results(results, output)
        percent = len([c for c in classes if c[0]]) / len(classes) * 100
        print("{0}% success".format(percent))
        with open('balanced_test_on_train_dense.txt', "a") as file:
            file.write("{0},{1}\n".format(epochs,percent))

    def check_results(self, results, output_t):
        return [(self.is_class_hit(results[i], output_t[i][0]), output_t[i][0], results[i]) for i in range(0, len(results))]

    def is_class_hit(self, result, real_result):
        resulting_class = 1 if result[1] > result[0] else 0
        return resulting_class == real_result


if __name__ == "__main__":
    network = SimpleKerasTest()
    for i in range(1, 400):
        network.learn_and_test_on_the_same_data('../balanced_results_random_very_easy.txt', i)
