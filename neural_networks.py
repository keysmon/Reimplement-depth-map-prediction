import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.keras as keras
import tensorflow as tf
tf.config.run_functions_eagerly(True)
import os
from data import import_data
import cv2
import numpy as np

# (2284, 320, 240, 3)
# (2284, 640, 480)
# (2284, 640, 480)

lambda_ = 0.5
cutoff = 100
def coarse_nn(X,Y):
    #print(X.shape)
    #print(Y.shape)
    model = keras.models.Sequential()
    
    # input was downsampled from the original by a factor of 2
    model.add(keras.layers.InputLayer(input_shape=(320,240,3)))
    model.add(keras.layers.Conv2D(96,(11,11),strides = (4,4),activation= 'relu',input_shape =(320,240,1),padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))
    
    model.add(keras.layers.Conv2D(256,(5,5),activation= 'relu',padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))
    
    model.add(keras.layers.Conv2D(384,(3,3),activation= 'relu',padding='same'))
    model.add(keras.layers.Conv2D(384,(3,3),activation= 'relu',padding='same'))
    model.add(keras.layers.Conv2D(384,(3,3),activation= 'relu',padding='same'))
    model.add(keras.layers.Flatten())
    
    model.add(keras.layers.Dense(4096,activation='relu'))
    model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.Dense(4800,activation='linear')) # output at 1/4 resolution of input
    model.add(keras.layers.Reshape((80,60)))
    
    model.compile(optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=0.0001),loss = Scale_invariant_loss,metrics = ['accuracy'],run_eagerly= False)
    model.fit(x = X,y = Y, epochs = 5 ,validation_split = 0.2)
    # print(model.layers[-1].output)
    
    return
    

def local_nn(X,Y):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(320,240,3)))

    model.add(keras.layers.Conv2D(63,(9,9),activation= 'relu',input_shape =(320,240,1),padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size = (2,2),strides = 4))
    
    
    #model.add(keras.layers.Concatenate(axis=1)([x,y]))
    
    model.add(keras.layers.Conv2D(64,(5,5),activation= 'relu',padding='same'))
    model.add(keras.layers.Conv2D(1,(5,5),activation= 'linear',padding='same'))
    model.compile(optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=0.1,momentum=0.9),loss = Scale_invariant_loss,metrics = ['accuracy'],run_eagerly= True)
    model.fit(x = X,y = Y, epochs = 5 ,batch_size = 32,validation_split = 0.2)
    
    
    return



def Scale_invariant_loss(y_true, y_pred):
    
    #n =y_true.shape[0]* y_true.shape[1] * y_true.shape[2]
    #tf.print(y_true.shape)
    #tf.print(y_pred.shape)
    
    #y_true = keras.backend.reshape(y_true,[-1])
    #y_pred = keras.backend.reshape(y_pred,[-1])
    #tf.print(y_true.shape)
    #tf.print(y_pred.shape)
    
    
    y_pred = tf.clip_by_value(y_pred,0,y_pred.dtype.max)
    
    #tf.print("y_pred_max",tf.math.reduce_max(y_pred))
    #tf.print("y_pred_min",tf.math.reduce_min(y_pred))
    
    log_y_true = keras.backend.log(y_true+keras.backend.epsilon())
    log_y_pred = keras.backend.log(y_pred+keras.backend.epsilon())
    diff_log = log_y_true - log_y_pred
    
    #tf.print("diff_log_max",tf.math.reduce_max(diff_log))
    #tf.print("diff_log_min",tf.math.reduce_min(diff_log))


    
    
    squared_diff_log = keras.backend.square(diff_log)
    #tf.print("square_diff_log_max",tf.math.reduce_max(squared_diff_log))
    #tf.print("square_diff_log_min",tf.math.reduce_min(squared_diff_log))
    loss_term1 = tf.math.reduce_mean(squared_diff_log)
    #loss_term1 = tf.math.reduce_sum(squared_diff_log)/keras.backend.get_value(tf.size(squared_diff_log))
    #tf.print(tf.size(squared_diff_log))
    
    #tf.print("loss1_before",loss_term1)
    #loss_term1 = loss_term1/(tf.size(squared_diff_log))
    #print(keras.backend.get_value(tf.size(squared_diff_log)))
    #tf.print("loss1_after",loss_term1)
    #loss_term1 = tf.math.reduce_mean(loss_term1,[1])

    loss_term2 = tf.math.reduce_sum(squared_diff_log)
    n = keras.backend.get_value(tf.size(squared_diff_log))
    loss_term2 = loss_term2/n
    loss_term2 = keras.backend.square(loss_term2)
   
    
    tf.print("loss1",loss_term1)
    tf.print("loss2",loss_term2)

    return loss_term1 + loss_term2*lambda_
    
    #return 0
def test():
    y_pred = tf.convert_to_tensor(np.array([[-1,0,3],[3,3,3],[3,3,3]]))
    y_true = tf.convert_to_tensor(np.array([[4,4,4],[4,4,4],[4,4,4]]))
    Scale_invariant_loss(y_true,y_pred)

def main():
    data_depth,data_images = import_data()
    data_depth = data_depth[:cutoff]
    data_images = data_images[:cutoff]
    coarse_nn(data_images,data_depth)
    #local_nn(data_images,data_depth)
    #test()
    return
    
if __name__ == "__main__":
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
