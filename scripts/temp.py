from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

r = tf.random_normal([10,3])
dataset = tf.data.Dataset.from_tensor_slices(r)
oneshot = dataset.make_one_shot_iterator()
iterator = dataset.make_initializable_iterator()
next_shot = nonshot.get_next()
next_iter = iterator.get_next()

sess = tf.Session()
sess.run(iterator.initializer)
while True:
  try:
    print(sess.run(next_shot, next_iter))
  except tf.errors.OutOfRangeError:
    break

