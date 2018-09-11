from matplotlib import pyplot

data_file = './keras/balanced_test_on_train_dense.txt'
# data_file = './keras/balanced_cross_validation_dense.txt'
# data_file = './keras/test_on_train_dense.txt'
# data_file = './keras/cross_validation_dense.txt'

# data_file = './keras/cross_validation_sparse_far.txt'

y = []
x = []
with open(data_file, 'r') as file:
    for line in file.readlines():
        content = line.split(',')
        x.append(int(content[0]))
        # y.append(content[1])
        y.append(float(content[1]))

pyplot.plot(x, y)
pyplot.axis([0, 400, 0, 100])
# pyplot.yticks([0, 10, 20, 40, 60])
pyplot.ylabel('Accuracy [%]')
pyplot.xlabel('Epochs')
pyplot.title('Tests on train data')
# pyplot.title('Tests on test data (with Cross-Validation)')
pyplot.show()
