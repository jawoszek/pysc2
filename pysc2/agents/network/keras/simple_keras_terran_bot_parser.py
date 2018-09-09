from keras.models import Sequential
from keras.layers import Dense, Dropout
from numpy import array
from pysc2.agents.network.terran_bot_parser import BotDataParser, BUILD_NODES, RECRUIT_NODES
from keras.utils import to_categorical

TOTAL_NUMBER_OF_BASE_NEURONS = BUILD_NODES * 2 + RECRUIT_NODES


class SimpleKerasTerranBotNetwork:

    def __init__(self) -> None:
        super().__init__()
        self.model = Sequential()
        self.model.add(Dense(units=64, activation='relu', input_dim=140))
        self.model.add(Dropout(0.5))
        # self.model.add(Dense(64, activation='relu'))
        # self.model.add(Dropout(0.5))
        self.model.add(Dense(units=2, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])

    def train_on_data_from_file(self, path):
        parser = BotDataParser()
        input, output = parser.read_data_file(path)

        x_train = array(input)
        y_train = to_categorical(array(output).T[0])

        self.model.fit(x_train, y_train, epochs=50000, batch_size=256)

    def test_data_from_file(self, path):
        parser = BotDataParser()
        input_t, output_t = parser.read_data_file(path)

        print(output_t)
        x_test = array(input_t)
        y_test = to_categorical(array(output_t).T[0])
        print(y_test)
        # loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
        results = self.model.predict(x_test, batch_size=256)
        return [(self.is_class_hit(results[i], output_t[i][0]), output_t[i][0], results[i]) for i in range(0, len(results))]

    def is_class_hit(self, result, real_result):
        resulting_class = 1 if result[1] > result[0] else 0
        return resulting_class == real_result


if __name__ == "__main__":
    network = SimpleKerasTerranBotNetwork()
    network.train_on_data_from_file('../balanced_results_random_very_easy.txt')
    classes = network.test_data_from_file('../tests_random_very_easy.txt')
    print(classes)
    print("{0}% success".format(len([c for c in classes if c[0]])/len(classes)*100))
