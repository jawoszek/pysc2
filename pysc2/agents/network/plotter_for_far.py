from matplotlib import pyplot

# data_file = './keras/balanced_test_on_train_sparse.txt'
data_file = './keras/balanced_cross_validation_sparse.txt'
# data_file = './keras/test_on_train_sparse.txt'
# data_file = './keras/cross_validation_sparse.txt'

# far_data_file = './keras/balanced_test_on_train_sparse_far.txt'
far_data_file = './keras/balanced_cross_validation_sparse_far.txt'
# far_data_file = './keras/test_on_train_sparse_far.txt'
# far_data_file = './keras/cross_validation_sparse_far.txt'

y = []
x = []
with open(data_file, 'r') as file:
    for line in file.readlines():
        content = line.split(',')
        x.append(int(content[0]))
        # y.append(content[1])
        y.append(float(content[1]))

f_y = []
f_x = []
with open(far_data_file, 'r') as file:
    for line in file.readlines():
        content = line.split(',')
        f_x.append(int(content[0]))
        # y.append(content[1])
        f_y.append(float(content[1]))

pyplot.plot(x, y)
pyplot.axis([0, 400, 0, 100])

print(f_x)
print(f_y)
# pyplot.plot(x, y, 'b')
pyplot.plot(x, y, 'b', f_x, f_y, 'r--')
# pyplot.yticks([0, 10, 20, 40, 60])
# pyplot.axis([0, 400, 0, 100])
pyplot.axis([0, 2000, 0, 100])
pyplot.ylabel('Accuracy [%]')
pyplot.xlabel('Epochs')
pyplot.title('Tests on train data')

pyplot.annotate('fine grained data', xy=(x[50], y[50]-4), xytext=(x[50]+ 200, y[50]-30),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
pyplot.annotate('approximate', xy=(f_x[7], f_y[7]), xytext=(f_x[10], f_y[10]-40),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
# pyplot.title('Tests on test data (with Cross-Validation)')
pyplot.show()
