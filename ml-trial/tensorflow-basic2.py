import tensorflow as tf

a = tf.Variable(1, name='a')
b = tf.constant(1, name='b')
c = tf.assign(a, a + b)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('1: [c, a] =', sess.run([c, a]))
    print('2: [c, a] =', sess.run([c, a]))