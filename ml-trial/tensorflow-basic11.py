import tensorflow as tf

# define parameters as variables.
x = tf.Variable(0., name='x')
# define the function to be minimized by parameters.
func = (x - 1) ** 2

# set the learning-rate
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=0.1
)

# train_step is the operation to move the values of x.
train_step = optimizer.minimize(func)

# execute train_step repeatedly.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        sess.run(train_step)
    print('x = ', sess.run(x))
