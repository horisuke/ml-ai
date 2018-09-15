import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 32

def get_batches(x, y, batch_size):
    n_data = len(x)
    indices = np.arange(n_data)
    np.random.shuffle(indices)
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    # Pick up (batch_sizes) data randomly from original data.
    for i in range(0, n_data, batch_size):
        x_batch = x_shuffled[i: i + batch_size]
        y_batch = y_shuffled[i: i + batch_size]
        yield x_batch, y_batch


# The followings are same as tensorflow-basic12.py ----------------
# Download the dataset of Boston house-prices
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()

# global settings of matplotlib
plt.rcParams['font.size'] = 10 * 3
plt.rcParams['figure.figsize'] = [18, 12]
plt.rcParams['font.family'] = ['IPAexGothic']

# display histogram "house prices/$1000" vs "number of data"
plt.hist(y_train, bins=20)  # bins:number of bar in histogram
plt.xlabel('house prices/$1000')
plt.ylabel('number of data')
plt.show()

# display plot "number of rooms" vs "house prices/$1000"
plt.plot(x_train[:, 5], y_train, 'o')
# 6th column is "number of rooms"
# 'o' means circle marker
plt.xlabel('number of rooms')
plt.ylabel('house prices/$1000')
plt.show()

# preprocessing
x_train_mean = x_train.mean(axis=0)
x_train_std = x_train.std(axis=0)
y_train_mean = y_train.mean()
y_train_std = y_train.std()

x_train = (x_train - x_train_mean) / x_train_std
y_train = (y_train - y_train_mean) / y_train_std
x_test = (x_test - x_train_mean) / x_train_std
y_test = (y_test - y_train_mean) / y_train_std

# display plot "number of rooms" vs "house prices/$1000" after preprocessing
plt.plot(x_train[:, 5], y_train, 'o')
plt.xlabel('number of rooms(after)')
plt.ylabel('house prices/$1000(after)')
plt.show()

# define the inference model of Boston house-prices
x = tf.placeholder(tf.float32, (None, 13), name='x')
y = tf.placeholder(tf.float32, (None, 1), name='y')
w = tf.Variable(tf.random_normal((13, 1)))
pred = tf.matmul(x, w)

# define the loss function and learning rate
loss = tf.reduce_mean((y - pred) ** 2)
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=0.1
)
train_step = optimizer.minimize(loss)
# The above are same as tensorflow-basic12.py ----------------

# execute train step and repeatedly
step = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(100):
        for x_batch, y_batch in get_batches(x_train, y_train, 32):
            train_loss, _ = sess.run(
                [loss, train_step],
                feed_dict={
                    x: x_batch,
                    y: y_batch.reshape((-1, 1))
                }
            )
            print('step: {}, train_loss: {}'.format(step, train_loss))
            step += 1

    pred_ = sess.run(
        pred,
        feed_dict={
            x: x_test
        }
    )