data_x = [0.5, 1.4, 2.1, 3.5, 4.2, 5.1, 6.3, 7.6, 8.2, 9.0]
data_y = [2.1, 3.5, 5.8, 7.1, 9.3, 11.5, 13.2, 16.0, 17.1, 19.2]

data_x_reshaped = [[x] for x in data_x]

train_x = [[x] for x in data_x[:6]]
train_y = data_y[:6]
train_x, train_y

test_x = [[x] for x in data_x[6:]]
test_y = data_y[6:]
test_x, test_y
