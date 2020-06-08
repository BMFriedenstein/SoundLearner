# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 13:04:38 2019

@author: Brandon
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from PIL import Image
from cnn_model_trainer import X,BuildModel


input_x = []
model_name = "tmp/model"
filename = "test_9992"
optimizer, loss, out = BuildModel()
saver = tf.train.Saver()
with tf.Session() as sess:
    pic = Image.open(filename + ".png").convert("RGB")
    input_x.append(np.array(pic,dtype= np.float16)/255)
    input_x = np.array(input_x, dtype=np.float16)
    saver.restore(sess, model_name)
    sess.run(out, feed_dict={X: input_x})
    print("Model restored.")
    np.savetxt(filename + "_out.data", out.eval(feed_dict={X: input_x})[0], delimiter=",", fmt="%.6f")