from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import os
import numpy as np
import tensorflow as tf
from PIL import Image
tf.reset_default_graph()
    
    
IMG_SIZE = 512
NUM_OUTPUTS = 50
NUM_EPOCH = 10001
BATCH_SIZE = 25
SET_SIZE = 4000
IMG_DEPTH = 2
FLAGS = None
X = tf.placeholder(tf.float16, shape=[None, IMG_SIZE,IMG_SIZE,IMG_DEPTH], name = "X")
Y = tf.placeholder(tf.float16, shape=[None,NUM_OUTPUTS,8], name = "Y")

restore = False
output_directory = "tmp"
summary_directory = "summary/"

def VariableSummaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
        
def WeightVariable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=np.float16)
    return tf.Variable(initial)

def BiasVariable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape, dtype=np.float16)
    return tf.Variable(initial)

def CreateLayer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
          weights = WeightVariable([input_dim, output_dim])
          VariableSummaries(weights)
        with tf.name_scope('biases'):
          biases = BiasVariable([output_dim])
          VariableSummaries(biases)
        with tf.name_scope('Wx_plus_b'):
          preactivate = tf.matmul(input_tensor, weights) + biases
          tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations

def BuildModel():
    """Model function for CNN."""
    global IMG_SIZE 
    global X 
    global Y 
    with tf.name_scope('input'):        
        X_reshaped = tf.reshape(X, shape=[-1,IMG_SIZE,IMG_SIZE,IMG_DEPTH])
    
	# 2x512x512
    with tf.name_scope('l1'):
        conv1 = tf.layers.conv2d(
                inputs=X_reshaped,
                filters=8,
                kernel_size=3,
                padding="same",
                activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
   
   # 8x256x256
    with tf.name_scope('l2'):
        conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=5,
                padding="same",
                activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
	# 64x128x128
    with tf.name_scope('l3'):
        conv3 = tf.layers.conv2d(
                inputs=pool2,
                filters=256,
                kernel_size=3,
                padding="same",
                activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    
	# 16x64x64
    with tf.name_scope('output_shaped'):
        output_layer = tf.layers.conv2d(
                        inputs=pool3,
                        filters=1,
                        kernel_size=[64-NUM_OUTPUTS+1,57],
                        padding="valid",
                        activation=tf.nn.relu)

    with tf.name_scope('train'):        
        with tf.name_scope('accuracy'):            
            average_loss = tf.losses.mean_squared_error(Y, output_layer[:,:,:,0])
            total_loss = tf.to_float(NUM_OUTPUTS * 8) * average_loss
            tf.summary.scalar('accuracy', average_loss)
            tf.summary.scalar('total_accuracy', total_loss)
        optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.001, momentum = 0.01)
        train_op = optimizer.minimize(loss=average_loss, global_step=tf.train.get_global_step())

    return train_op, total_loss, output_layer[:,:,:,0]

def load_image(filename_tensor):
    return tf.image.decode_png(filename_tensor)

train_images = []
train_labels = []

def GetDataset():
    global train_images
    global train_labels
    global test_images
    global test_labels
    print("Loading input of " + str(SET_SIZE) + " images")
    count = 0
    for filename in os.listdir("img/"):
        if (count%100 is 0):
            print(str(count) + " of " + str(SET_SIZE) + " " + str(count*IMG_SIZE*IMG_SIZE*IMG_DEPTH*16/8000000) + " MB")
        if (count >= SET_SIZE):
            break
        pic = Image.open("img/"+filename).convert("RGB")
        pix = np.array(pic,dtype= np.float16)/255
        
        if(IMG_DEPTH is 2):
           pix = np.delete(pix,1, axis=2)
        train_images.append(pix)
        count += 1
    count = 0
    
    print("Loading labels of " + str(SET_SIZE) + " images")
    for filename in os.listdir("output/"):
        if (count%100 is 0):
            print(str(count) + " of " + str(SET_SIZE))
        if (count >= SET_SIZE):
            break
        with open("output/" + filename, newline='') as f:
            rows = []
            reader = csv.reader(f, delimiter = ',')
            for row in reader:
                row = [np.float16(i) for i in row]
                rows.append(row)
                train_labels.append(rows)
        count += 1    
    train_images = np.array(train_images, dtype=np.float16)
    train_labels = np.array(train_labels, dtype=np.float16)

if __name__ == "__main__":
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    optimizer, loss, out = BuildModel()
    GetDataset()
    counter = 0
    
    with tf.name_scope("init_and_save"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()  
         
    with tf.device("/gpu:0"):
        sess = tf.Session()
        init.run(session=sess)
        
    if(restore):
        saver.restore(sess, output_directory +"/model")
        
    train_writer = tf.summary.FileWriter( summary_directory, sess.graph)
    for epoch in range(NUM_EPOCH):
        for iteration in range(SET_SIZE//BATCH_SIZE):
            x_batch = train_images[iteration*BATCH_SIZE:iteration*BATCH_SIZE+BATCH_SIZE,:]
            y_batch = train_labels[iteration*BATCH_SIZE:iteration*BATCH_SIZE+BATCH_SIZE,:]
            merge = tf.summary.merge_all()
            summary, totalloss, _ = sess.run([merge,loss,optimizer], feed_dict={X: x_batch, Y: y_batch})
            counter += 1
        print("Epoch: " + str(epoch+1) + " loss: " + str(totalloss))		
        train_writer.add_summary(summary, counter)
        if( epoch%25 is 0 ):
            save_path = saver.save(sess, output_directory + "/model")
   