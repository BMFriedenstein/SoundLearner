from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import os
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import read
from scipy.fftpack import fft
tf.reset_default_graph()
INPUT_SIZE = 4*44100
NUM_OUTPUTS = 1
NUM_EPOCH = 200
BATCH_SIZE = 25
SET_SIZE = 2000
META_SIZE = 1000
INPUT_DEPTH = 1
FLAGS = None
restore = False

X = tf.placeholder(tf.float16, shape=[None, INPUT_SIZE, INPUT_DEPTH], name = "X")
M = tf.placeholder(tf.float16, shape=[None, 3], name = "M")
Y = tf.placeholder(tf.float16, shape=[None, NUM_OUTPUTS, 8], name = "Y")
K = tf.placeholder(tf.float16, name = "K")

output_directory = "tmp"
def BuildModel():
    global INPUT_SIZE, Y, X, K
    
    X_reshaped = tf.reshape(X, shape=[-1,INPUT_SIZE,INPUT_DEPTH])
    
	# 1x176400
    conv1 = tf.layers.conv1d(
            inputs=X_reshaped,
            filters=24,
            kernel_size=32,
            padding="same",
            activation=tf.nn.relu)     
    dropconv1 = tf.nn.dropout(conv1, keep_prob=K)           
    pool1 = tf.layers.max_pooling1d(inputs=dropconv1, pool_size=4, strides=4)
        
    # 44100*24
    conv2 = tf.layers.conv1d(
            inputs=pool1,
            filters=32,
            kernel_size=32,
            padding="same",
            activation=tf.nn.relu)
    dropconv2 = tf.nn.dropout(conv2, keep_prob=K)  
    pool2 = tf.layers.max_pooling1d(inputs=dropconv2, pool_size=4, strides=4)
    
    # 8x11025
    conv4 = tf.layers.conv1d(
            inputs=pool2,
            filters=64,
            kernel_size=64,
            padding="same",
            activation=tf.nn.relu)
    dropconv4 = tf.nn.dropout(conv4, keep_prob=K)  
    pool4 = tf.layers.max_pooling1d(inputs=dropconv4, pool_size=4, strides=4)
        
    # 64x2756
    conv5 = tf.layers.conv1d(
            inputs=pool4,
            filters=128,
            kernel_size=128,
            padding="same",
            activation=tf.nn.relu)
    dropconv5 = tf.nn.dropout(conv5, keep_prob=K)  
    pool5 = tf.layers.max_pooling1d(inputs=dropconv5, pool_size=2, strides=2)
       
     # 256x1378  
    conv6 = tf.layers.conv1d(
            inputs=pool5,
            filters=256,
            kernel_size=128,
            padding="same",
            activation=tf.nn.relu)
    dropconv6 = tf.nn.dropout(conv6, keep_prob=K)  
    pool6 = tf.layers.max_pooling1d(inputs=dropconv6, pool_size=2, strides=2)
        
    # 256x689  
    conv7 = tf.layers.conv1d(
            inputs=pool6,
            filters=256,
            kernel_size=128,
            padding="same",
            activation=tf.nn.relu)
    dropconv7 = tf.nn.dropout(conv7, keep_prob=K)  
    pool7 = tf.layers.max_pooling1d(inputs=dropconv7, pool_size=2, strides=2)
    
    # 256x344
    conv8 = tf.layers.conv1d(
            inputs=pool7,
            filters=256,
            kernel_size=128,
            padding="same",
            activation=tf.nn.relu)
    dropconv8 = tf.nn.dropout(conv8, keep_prob=K)  
    pool8 = tf.layers.max_pooling1d(inputs=dropconv8, pool_size=2, strides=2)
    
    # 256x177
    output_layer = tf.layers.conv1d(
            inputs=pool8,
            filters=NUM_OUTPUTS*2,
            kernel_size=169,
            padding="valid",
            activation=tf.nn.relu)    
    
    with tf.name_scope('output_shaped'):
        output_layer_reshape = tf.reshape(output_layer, [-1, NUM_OUTPUTS,8])
        
    with tf.name_scope('train'):        
        with tf.name_scope('accuracy'):            
            average_loss = tf.losses.mean_squared_error(Y, output_layer_reshape)
            total_loss = tf.to_float(NUM_OUTPUTS * 8 * average_loss) 
        optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.01, momentum = 0.1)
        train_op = optimizer.minimize(loss=average_loss, global_step=tf.train.get_global_step())

    return train_op, total_loss, output_layer_reshape

def load_image(filename_tensor):
    return tf.image.decode_png(filename_tensor)

train_input = []
train_labels = []
train_scalers = []

def GetDataset():
    global train_input
    global train_labels
    global test_images
    global test_labels
    global train_scalers
    print("Loading input of " + str(SET_SIZE) + " wave files")
    count = 0
   
    for filename in os.listdir("wav/"):
        if (count%100 is 0):
            print(str(count) + " of " + str(SET_SIZE) + " " + str(count*INPUT_SIZE*INPUT_DEPTH*16/8000000) + " MB")
        if (count >= SET_SIZE):
            break
        scaler = []
        #split_input = []
        wav_file = read("wav/"+filename)
        wave_data = np.float16(wav_file[1])/32767
        #wave_data = np.split(wave_data, 2)
                
       # split_input.append(wave_data[0])
        #split_input.append(wave_data[1])        
        #fft0 = np.abs(fft(wave_data[0]))
        #fft1 = np.abs(fft(wave_data[1]))
        #split_input.append(fft0/np.max(fft0))
        #split_input.append(fft1/np.max(fft1))
        
        with open("meta/" + filename.replace("wav","meta"), newline='') as f:
            rows = []
            reader = csv.reader(f, delimiter = ',')
            row_n = 0
            for row in reader:
                row = [np.float16(i) for i in row]
                if(row_n is 2): 
                    #sustain_data =  np.split( np.array(row), 2) 
                    #split_input.append(np.array(row))
                    #split_input.append(sustain_data[1])
                    2+1
                elif(row_n is 0):
                    scaler.append(row[0]/20000)
                    scaler.append(row[1]/20000)
                else:                    
                    scaler.append(row[0])
                row_n += 1
        train_scalers.append(scaler)
        train_input.append(np.array([wave_data]).transpose())
        count += 1
    count = 0

    print("Loading labels of " + str(SET_SIZE) + " FILES")
    for filename in os.listdir("data"):
        if (count%100 is 0):
            print(str(count) + " of " + str(SET_SIZE))
        if (count >= SET_SIZE):
            break
        with open("data/" + filename, newline='') as f:
            rows = []
            reader = csv.reader(f, delimiter = ',')
            for row in reader:
                row = [np.float16(i) for i in row]
                rows.append(row)
            train_labels.append(rows)
        count += 1    
    train_input = np.array(train_input, dtype=np.float16)
    train_scalers = np.array(train_scalers, dtype=np.float16)
    train_labels = np.array(train_labels, dtype=np.float16)

if __name__ == "__main__":
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    optimizer, loss, out = BuildModel()
    
    counter = 0
    
    with tf.name_scope("init_and_save"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()  
        
    with tf.device("/gpu:0"):
        sess = tf.Session()
        init.run(session=sess)
        
    if(restore):
        saver.restore(sess, output_directory +"/model")
    GetDataset()
    last_best = 1000000
    for epoch in range(NUM_EPOCH):
        ave_loss=0
        for iteration in range(SET_SIZE//BATCH_SIZE):
            x_batch = train_input[iteration*BATCH_SIZE:iteration*BATCH_SIZE+BATCH_SIZE,:]
            y_batch = train_labels[iteration*BATCH_SIZE:iteration*BATCH_SIZE+BATCH_SIZE,:]
            m_batch = train_scalers[iteration*BATCH_SIZE:iteration*BATCH_SIZE+BATCH_SIZE,:]
            totalloss, _ = sess.run([loss,optimizer], feed_dict={X: x_batch, Y: y_batch, M: m_batch, K: 0.5})
            counter += 1
            ave_loss+=totalloss
        ave_loss = ave_loss/(SET_SIZE//BATCH_SIZE)
        print("Epoch: " + str(epoch+1) + " average loss: " + str(ave_loss))
        if(ave_loss < last_best or epoch%25 is 0 ):           
            last_best = ave_loss
            print("Saving model: ")
            save_path = saver.save(sess, output_directory +"/model")
   