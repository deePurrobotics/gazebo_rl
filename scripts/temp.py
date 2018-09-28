from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

# tf.enable_eager_execution()

max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
  # Initialize an iterator over a dataset with 10 elements.
  sess.run(iterator.initializer, feed_dict={max_value: 10})
  for i in range(10):
    value = sess.run(next_element)
    print(value)
    assert i == value

  # Initialize the same iterator over a dataset with 100 elements.
  sess.run(iterator.initializer, feed_dict={max_value: 100})
  for i in range(100):
    value = sess.run(next_element)
    print(value)
    assert i == value
