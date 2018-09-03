import tensorflow as tf

a = tf.Variable(1, name='a')
b = tf.assign(a, a+1)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b))
    print(sess.run(b))
    # Save the values of variables to model/model.ckpt.
    saver.save(sess, 'model/model.ckpt')

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Restore the values of variables from model/model.ckpt.
    saver.restore(sess, save_path='model/model.ckpt')
    print(sess.run(b))
    print(sess.run(b))