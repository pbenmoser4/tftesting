import tensorflow as tf
from os.path import join

LOGDIR = './tf_logs/'

# Define nodes in the graph
x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')
mult_constant = tf.constant(4.0, name='c')

# defining the operation nodes
f = x*x*y + mult_constant*y

# Create a session object
sess = tf.Session()

# run the nodes that produce the results
print(sess.run(f, feed_dict={x:3, y:2}))

writer = tf.summary.FileWriter(join(LOGDIR, 'simple'), sess.graph)

writer.close()
sess.close()
