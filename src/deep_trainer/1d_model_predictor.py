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
from scipy.io.wavfile import read
from scipy.fftpack import fft
from cnn_1d_model_trainer import K,X,M,BuildModel
import csv


input_m = []
input_x = []

model_name = "small_batch_1000"
filename = "vio2"
optimizer, loss, out = BuildModel()
saver = tf.train.Saver()
with tf.Session() as sess: 
    split_input = []
    scaler = []
    wav_file = read(filename+".wav")
    wave_data = np.float16(wav_file[1])/32767
    wave_data = np.split(wave_data, 2) 
    split_input.append(wave_data[0])
#    split_input.append(wave_data[1])        
    fft0 = np.abs(fft(wave_data[0]))
#    fft1 = np.abs(fft(wave_data[1]))
    split_input.append(fft0/np.max(fft0))
#    split_input.append(fft1/np.max(fft1))  
    
    with open("default.meta", newline='') as f:
        rows = []
        reader = csv.reader(f, delimiter = ',')
        row_n = 0
        for row in reader:
            row = [np.float16(i) for i in row]
            if(row_n is 2): 
 #               split_input.append(np.array(row)) 
               1
            elif(row_n is 0):
                scaler.append(row[0]/20000)
                scaler.append(row[1]/20000)
            else:                    
                scaler.append(row[0])
            row_n += 1
    input_x.append(np.array(split_input).transpose())
    input_m.append(np.array(scaler))
    saver.restore(sess, model_name + "/model")
    sess.run(out,feed_dict={X: input_x, M: input_m, K: 1.0})
    print("Model restored.")
    np.savetxt(filename + "_" + model_name +  "_out.data", out.eval(feed_dict={X: input_x, M: input_m, K: 1.0})[0], delimiter=",", fmt="%.6f")