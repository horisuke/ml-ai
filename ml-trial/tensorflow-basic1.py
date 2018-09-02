import tensorflow as tf

a = tf.constant(1, name='a')
b = tf.constant(2, name='b')
c = a + b
graph = tf.get_default_graph()

with tf.Session() as sess:
    print(c)
    print(sess.run(c))
    print(graph.as_graph_def())

