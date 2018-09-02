import tensorflow as tf

a = tf.constant([1,2,3], name='a')
b = tf.constant([4,5,6], name='b')
c = a + b

with tf.Session() as sess:
    print('a + b = ', sess.run(c))

