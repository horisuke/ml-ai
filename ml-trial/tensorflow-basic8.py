import tensorflow as tf

a = tf.Variable(1, name='a')
b = tf.assign(a, a+1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('1st b = ', sess.run(b))
    print('2nd b = ', sess.run(b))

# session is changed.
with tf.Session() as sess:
    print('-- New session --')
    sess.run(tf.global_variables_initializer())
    print('1st b = ', sess.run(b))
    print('2nd b = ', sess.run(b))