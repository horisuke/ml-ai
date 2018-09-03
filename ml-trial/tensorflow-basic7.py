import tensorflow as tf

a = tf.placeholder(shape=(None, 2), dtype=tf.int32, name='a')

with tf.Session() as sess:
    print('-- Insert [[1, 2]] --')
    print('a = ', sess.run(a, feed_dict={a:[[1,2]]}))
    print('\n-- Insert [[1, 2], [3, 4]] --')
    print('a = ', sess.run(a, feed_dict={a: [[1, 2], [3, 4]]}))
