import tensorflow as tf

a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
c = tf.add(a, b)
d = tf.multiply(a, b)

with tf.Session() as sess:
    print('a + b = ', sess.run(c))
    print('a * b = ', sess.run(d))

