import tensorflow.keras as keras
import tensorflow as tf
import os
from data import import_data
import cv2
import numpy as np

# (2284, 320, 240, 3)
# (2284, 640, 480)
# (2284, 640, 480)

lambda_ = 0.5

def coarse_nn(X,Y):
    print(X.shape)
    print(Y.shape)
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
    
    model.compile(optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=0.1,momentum=0.9),loss = Scale_invariant_loss,metrics = ['accuracy'],run_eagerly= False)
    model.fit(x = X,y = Y, epochs = 30 ,batch_size = 32,validation_split = 0.2)
    return
    
'''
def local_nn(X,Y,coarse_output):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(320,240,3)))
    model.add(keras.layers.Conv2D(63,(9,9),activation= 'relu',input_shape =(320,240,1),padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size = (2,2),strides = 4))
    return
'''


def Scale_invariant_loss(y_true, y_pred):
    n = y_true.shape[0]
    diff_log = tf.math.log(y_true+1e-4) - tf.math.log(y_pred+1e-4)
    #print(diff_log)
    
    #print(type(diff_log))
    # Convert any value of nan caused by log(x) where x<= 0 to 0
    value_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(diff_log)), dtype=tf.float32) # inspired by https://stackoverflow.com/questions/42043488/replace-nan-values-in-tensorflow-tensor to resolve loss resulting in nan
    diff_log = tf.math.multiply_no_nan(diff_log, value_not_nan)
    #print(diff_log)
    
    
    squared_diff_log = tf.math.square(diff_log)
    #print(squared_diff_log)
    loss_term1 = tf.keras.backend.mean(squared_diff_log)
    loss_term2 = tf.math.square(tf.keras.backend.mean(diff_log))*lambda_
    #print(loss_term1)
    #print(loss_term2)
    return loss_term1-loss_term2

def test():
    y_pred = np.array([[-1,0,3],[3,3,3],[3,3,3]])
    y_true = np.array([[4,4,4],[4,4,4],[4,4,4]])
    Scale_invariant_loss(y_true,y_pred)

def main():
    data_depth,data_images = import_data()
    coarse_nn(data_images,data_depth)
    #test()
    return
    
if __name__ == "__main__":
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
