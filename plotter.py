from data import visualize_depth_map
#import tensorflow.keras as keras
#import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_input():
    npzfile = np.load('input_data.npz')
    data_depth = npzfile['x']
    data_images = npzfile['y']
    return data_images,data_depth
def load_coarse_models():
    coarse_model = keras.models.load_model("my_coarse_model", compile = False)

    return coarse_model

def load_fine_models():
    coarse_model = keras.models.load_model("my_fine_model", compile = False)

    return coarse_model
    
def model_predict(model,X):
    y = model.predict(X)
    return y

def plot_coarse():
    data_images,data_depth = load_input()
    coarse_model = load_coarse_models()
    coarse_prediction = model_predict(coarse_model,data_images)
    visualize_depth_map((data_images,coarse_prediction))
    #print(data_depth[0])
    #print(coarse_prediction[0])
    return

def plot_fine():
    data_images,data_depth = load_input()
    data_images = np.array([cv2.rotate(i,cv2.ROTATE_90_CLOCKWISE) for i in data_images])
    data_depth = np.array([cv2.rotate(i,cv2.ROTATE_90_CLOCKWISE) for i in data_depth])
    #fine_model = load_fine_models()
    #fine_prediction = model_predict(fine_model,data_images)
    visualize_depth_map((data_images,data_depth))
    #print(data_depth[0])
    #print(coarse_prediction[0])
    return 
    
def main():
    #plot_coarse()
    plot_fine()
    return
    
    
if __name__ == "__main__":
    main()