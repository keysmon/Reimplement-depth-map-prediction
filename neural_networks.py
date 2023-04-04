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
cutoff = 2284

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
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4800,activation='linear')) # output at 1/4 resolution of input
    model.add(keras.layers.Reshape((80,60)))
    
    model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=0.01),loss = Scale_invariant_loss,metrics = ['accuracy'],run_eagerly = False)
    model.fit(x = X,y = Y, epochs = 50,batch_size = 32)
    #coarse_output = model.predict(X)
    #model.save("my_log_model")
    
    return model



def fine_net(X,Y,coarse_model):
    input_1 = keras.layers.Input(shape=(320,240,3))
    layer_1_1 = keras.layers.Conv2D(96,(11,11),strides = (4,4),activation= 'relu',input_shape =(320,240,1),padding='same')(input_1)
    #layer_1_1.trainable = False
    #layer_1_1.set_weights(coarse_model[0].get_weights())
     
     
    layer_1_2 = keras.layers.MaxPooling2D(pool_size = (2,2))(layer_1_1)
    
    layer_1_3 = keras.layers.Conv2D(256,(5,5),activation= 'relu',padding='same')(layer_1_2)
    #layer_1_3.trainable = False
    #layer_1_3.set_weights(coarse_model[2].get_weights())
    
    layer_1_4 = keras.layers.MaxPooling2D(pool_size = (2,2))(layer_1_3)
    
    layer_1_5 = keras.layers.Conv2D(384,(3,3),activation= 'relu',padding='same')(layer_1_4)
    #layer_1_5.trainable = False
    #layer_1_5.set_weights(coarse_model[4].get_weights())
    
    layer_1_6 = keras.layers.Conv2D(384,(3,3),activation= 'relu',padding='same')(layer_1_5)
    #layer_1_6.trainable = False
    #layer_1_6.set_weights(coarse_model[5].get_weights())
    
    layer_1_7 = keras.layers.Conv2D(384,(3,3),activation= 'relu',padding='same')(layer_1_6)
    #layer_1_7.trainable = False
    #layer_1_7.set_weights(coarse_model[6].get_weights())
    
    layer_1_8 = keras.layers.Flatten()(layer_1_7)
    
    layer_1_9 = keras.layers.Dense(4096,activation='relu')(layer_1_8)
    #layer_1_9.trainable = False
    #layer_1_9.set_weights(coarse_model[8].get_weights())
    
    layer_1_10 = keras.layers.Dropout(0.5)(layer_1_9)
    layer_1_11 = keras.layers.Dense(4800,activation='linear')(layer_1_10)
    #layer_1_11.trainable = False
    #layer_1_11.set_weights(coarse_model[10].get_weights())
    
    layer_1_12 = keras.layers.Reshape((80,60,1))(layer_1_11)
    
    layer_2_1 = keras.layers.Conv2D(63,(9,9),activation= 'relu',input_shape =(320,240,1),padding='same')(input_1)
    layer_2_2 = keras.layers.MaxPooling2D(pool_size = (2,2),strides = 4)(layer_2_1)
    layer_2_3 = keras.layers.Concatenate()([layer_1_12,layer_2_2])
    
    layer_2_4 = keras.layers.Conv2D(64,(5,5),activation= 'relu',padding='same')(layer_2_3)
    layer_2_5 = keras.layers.Conv2D(1,(5,5),activation= 'linear',padding='same')(layer_2_4)
    
    model = keras.Model(inputs = [input_1],outputs = [layer_2_5])
    model.compile(optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=0.01),loss = Scale_invariant_loss,metrics = ['accuracy'])

    #print(model.summary(show_trainable = True,expand_nested = True))
    #print(model.get_layer(index = 0))
    model.layers[1].trainable = False
    model.layers[3].trainable = False
    model.layers[5].trainable = False
    model.layers[6].trainable = False
    model.layers[7].trainable = False
    model.layers[9].trainable = False
    model.layers[11].trainable = False
    model.layers[1].set_weights(coarse_model.layers[0].get_weights())
    model.layers[3].set_weights(coarse_model.layers[2].get_weights())
    model.layers[5].set_weights(coarse_model.layers[4].get_weights())
    model.layers[6].set_weights(coarse_model.layers[5].get_weights())
    model.layers[7].set_weights(coarse_model.layers[6].get_weights())
    model.layers[9].set_weights(coarse_model.layers[8].get_weights())
    model.layers[11].set_weights(coarse_model.layers[10].get_weights())

    
    
    #print(model.layers[0])
    #print(coarse_model.layers[1])

    #model.layers[0].set_weights(coarse_model.layers[1].get_weights())
    print(model.summary(show_trainable = True,expand_nested = True))
    #print(coarse_model.summary(show_trainable = True,expand_nested = True))

    model.fit(x = X,y = Y, epochs = 50  ,batch_size = 32,validation_split=0.2)
    model.save("my_model")
    
    
    return
def Scale_invariant_loss(y_true, y_pred):
    #tf.print("\ny_true",y_true.shape)
    #tf.print("y_pred",y_pred.shape)
    
    y_pred = tf.clip_by_value(y_pred,0,y_pred.dtype.max)
    #tf.print("y_pred_max",keras.backend.max(y_pred))
    #tf.print("y_pred_min",keras.backend.min(y_pred))
    log_y_true = keras.backend.log(y_true+keras.backend.epsilon())
    log_y_pred = keras.backend.log(y_pred+keras.backend.epsilon())
   
    diff = log_y_pred - log_y_true  
    #tf.print("diff_max",keras.backend.max(diff))
    #tf.print("diff_min",keras.backend.min(diff))
    n = y_true.shape[1] * y_true.shape[2]
    

    squared_diff = keras.backend.square(diff)
    #tf.print("squared_diff_max",keras.backend.max(squared_diff))
    #tf.print("squared_diff_min",keras.backend.min(squared_diff))
    
    loss_term1 = tf.math.reduce_mean(squared_diff,[1,2])
    #tf.print("loss_term1",loss_term1.shape)
    #tf.print("loss_term1_max",keras.backend.max(loss_term1))
    #tf.print("loss_term1_min",keras.backend.min(loss_term1))
    
    loss_term2 = tf.math.reduce_sum(diff,[1,2])
    #tf.print("Reduced_sum",loss_term2)
    loss_term2 = keras.backend.square(loss_term2)
    #tf.print("Reduced_sum_squared",loss_term2)

    loss_term2 = 0.5*loss_term2/(n**2)
    #tf.print("loss_term2",loss_term2.shape)
    #tf.print("loss_term2_max",keras.backend.max(loss_term2))
    #tf.print("loss_term2_min",keras.backend.min(loss_term2))
    
    loss = loss_term1 - loss_term2
    #tf.print("per_sample_loss",loss)

    loss = tf.math.reduce_mean(loss)
    #tf.print("total_loss_mean",loss)

    return loss


def load_coarse(input):
    model = keras.models.load_model("my_model", compile = False)
    return model


def main():
    data_depth,data_images = import_data()
    #data_depth = data_depth[:cutoff]
    #data_images = data_images[:cutoff]
    coarse_output = coarse_nn(data_images,data_depth)
    coarse_model = load_coarse(data_images)
    fine_net(data_images,data_depth,coarse_model)
    
    return
 
if __name__ == "__main__":
    main()


'''
def local_nn(X,Y,coarse_output):
    input_2 = keras.layers.Input(shape=(320,240,3))
    layer2_1 = keras.layers.Conv2D(63,(9,9),activation= 'relu',input_shape =(320,240,1),padding='same')(input_2)
    layer2_2 = keras.layers.MaxPooling2D(pool_size = (2,2),strides = 4)(layer2_1)
    
    layer2_3 = keras.layers.Concatenate()([coarse_output,layer2_2])
    
    layer2_4 = keras.layers.Conv2D(64,(5,5),activation= 'relu',padding='same')(layer2_4)
    layer2_5 = keras.layers.Conv2D(1,(5,5),activation= 'linear',padding='same')(layer2_5)
    model = keras.models.Model(input=[input_2],outputs = [layer2_5])
    model.compile(optimizer =  tf.keras.optimizers.experimental.SGD(learning_rate=0.01,momentum=0.9),loss = Scale_invariant_loss,metrics = ['accuracy'],run_eagerly= True)
    model.fit(x = X,y = Y, epochs = 5)
    
    return

def Scale_invariant_loss(y_true, y_pred):
    #tf.print("y_true",y_true.shape)
    #tf.print("y_pred",y_pred.shape)
    
    y_pred = tf.clip_by_value(y_pred,0,y_pred.dtype.max)
    
    log_y_true = keras.backend.log(y_true+keras.backend.epsilon())
    log_y_pred = keras.backend.log(y_pred+keras.backend.epsilon())
    diff_log = log_y_pred - log_y_true


    squared_diff_log = keras.backend.square(diff_log)
 
    loss_term1 = tf.math.reduce_mean(squared_diff_log)
    loss_term2 = tf.math.reduce_sum(squared_diff_log)
    n = keras.backend.get_value(tf.size(squared_diff_log))
    loss_term2 = loss_term2/n
    loss_term2 = keras.backend.square(loss_term2)
   

    return loss_term1 - loss_term2*lambda_
'''