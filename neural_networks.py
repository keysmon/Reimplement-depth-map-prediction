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
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(4800,activation='linear')) # output at 1/4 resolution of input
    model.add(keras.layers.Reshape((80,60)))
    
    model.compile(optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=0.1),loss = Scale_invariant_loss,metrics = ['accuracy'],run_eagerly= False)
    model.fit(x = X,y = Y, epochs = 5  ,batch_size = 32)
    #coarse_output = model.predict(X)
    model.save("my_model")
    
    return coarse_output
    

def coarse_nn(X,Y):
    #print(X.shape)
    #print(Y.shape)
    63 x 80 x 60 + 80 x 60 = 64 x 80 x 60
    # input was downsampled from the original by a factor of 2
    input_1 = keras.layers.Input(shape=(320,240,3))
    layer_1_1 = keras.layers.Conv2D(96,(11,11),strides = (4,4),activation= 'relu',input_shape =(320,240,1),padding='same')(input_1)
    layer_1_2 = keras.layers.MaxPooling2D(pool_size = (2,2))(layer_1_1)
    layer_1_3 = keras.layers.Conv2D(256,(5,5),activation= 'relu',padding='same')(layer_1_2)
    layer_1_4 = keras.layers.MaxPooling2D(pool_size = (2,2))(layer_1_3)
    layer_1_5 = keras.layers.Conv2D(384,(3,3),activation= 'relu',padding='same')(layer_1_4)
    layer_1_6 = keras.layers.Conv2D(384,(3,3),activation= 'relu',padding='same')(layer_1_5)
    layer_1_7 = keras.layers.Conv2D(384,(3,3),activation= 'relu',padding='same')(layer_1_6)
    layer_1_8 = keras.layers.Flatten()(layer_1_7)
    layer_1_9 = keras.layers.Dense(4096,activation='relu')(layer_1_8)
    layer_1_10 = keras.layers.Dropout(0.1)(layer_1_9)
    layer_1_11 = keras.layers.Dense(4800,activation='linear')(layer_1_10)
    layer_1_12 = keras.layers.Reshape((80,60,1))(layer_1_11)
    
    
    
    layer_2_1 = keras.layers.Conv2D(63,(9,9),activation= 'relu',input_shape =(320,240,1),padding='same')(input_1)
    layer_2_2 = keras.layers.MaxPooling2D(pool_size = (2,2),strides = 4)(layer_2_1)
    layer_2_3 = keras.layers.Concatenate()([layer_1_12,layer_2_2])
    
    layer_2_4 = keras.layers.Conv2D(64,(5,5),activation= 'relu',padding='same')(layer_2_3)
    layer_2_5 = keras.layers.Conv2D(1,(5,5),activation= 'linear',padding='same')(layer_2_4)
    
    
    model = keras.Model(inputs = [input_1],outputs = [layer_2_5])
    model.compile(optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=0.01),loss = Scale_invariant_loss,metrics = ['accuracy'])
    model.fit(x = X,y = Y, epochs = 5  ,batch_size = 10)
    #coarse_output = model.predict(X)
    model.save("my_model")
    
    return

def coarse_nn_predict(input):
    model = keras.models.load_model("my_model", compile = False)
    coarse_output = model.predict(input)
    coarse_output = tf.convert_to_tensor(coarse_output)
    return coarse_output



def local_nn(X,Y,coarse_output):
    input_2 = keras.layers.Input(shape=(320,240,3))
    layer2_1 = keras.layers.Conv2D(63,(9,9),activation= 'relu',input_shape =(320,240,1),padding='same')(input_2)
    layer2_2 = keras.layers.MaxPooling2D(pool_size = (2,2),strides = 4)(layer2_1)
    
    layer2_3 = keras.layers.Concatenate()([coarse_output,layer2_2])
    
    layer2_4 = keras.layers.Conv2D(64,(5,5),activation= 'relu',padding='same')(layer2_4)
    layer2_5 = keras.layers.Conv2D(1,(5,5),activation= 'linear',padding='same')(layer2_5)
    model = keras.models.Model(input=[input_2],outputs = [layer2_5])
    model.compile(optimizer =  tf.keras.optimizers.experimental.SGD(learning_rate=0.1,momentum=0.9),loss = Scale_invariant_loss,metrics = ['accuracy'],run_eagerly= True)
    model.fit(x = X,y = Y, epochs = 5)
    
    return



def Scale_invariant_loss(y_true, y_pred):
    #tf.print("y_true",y_true.shape)
    #tf.print("y_pred",y_pred.shape)
    
    y_pred = tf.clip_by_value(y_pred,0,y_pred.dtype.max)
    
    log_y_true = keras.backend.log(y_true+keras.backend.epsilon())
    log_y_pred = keras.backend.log(y_pred+keras.backend.epsilon())
    diff_log = log_y_true - log_y_pred

    
    squared_diff_log = keras.backend.square(diff_log)
 
    loss_term1 = tf.math.reduce_mean(squared_diff_log)
    loss_term2 = tf.math.reduce_sum(squared_diff_log)
    n = keras.backend.get_value(tf.size(squared_diff_log))
    loss_term2 = loss_term2/n
    loss_term2 = keras.backend.square(loss_term2)
   


    return loss_term1 + loss_term2*lambda_
    



def main():
    data_depth,data_images = import_data()
    data_depth = data_depth[:cutoff]
    data_images = data_images[:cutoff]
    coarse_output = coarse_nn(data_images,data_depth)
    #coarse_output = coarse_nn_predict(data_images)
    #local_nn(data_images,data_depth,coarse_output)
    
    return
    
if __name__ == "__main__":
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
