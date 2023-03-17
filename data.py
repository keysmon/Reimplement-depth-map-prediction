import h5py
import numpy as np
import os
# accelData – Nx4 matrix of accelerometer values indicated when each frame was taken. The columns contain the roll, yaw, pitch and tilt angle of the device.
# depths – HxWxN matrix of depth maps where H and W are the height and width, respectively and N is the number of images. The values of the depth elements are in meters.
# images – HxWx3xN matrix of RGB images where H and W are the height and width, respectively, and N is the number of images.
# labels – HxWxN matrix of label masks where H and W are the height and width, respectively and N is the number of images. The labels range from 1..C where C is the total number of classes. If a pixel’s label value is 0, then that pixel is ‘unlabeled’.
# names – Cx1 cell array of the english names of each class.
# namesToIds – map from english label names to IDs (with C key-value pairs)
# rawDepths – HxWxN matrix of depth maps where H and W are the height and width, respectively, and N is the number of images. These depth maps are the raw output from the kinect.
# scenes – Cx1 cell array of the name of the scene from which each image was taken.

def import_data():
    if not os.path.exists('input_data.npy'):
        f = h5py.File('nyu_depth_data_labeled.mat', 'r')
        
        data_depth = f['depths']
        data_images = f['images']
        data_rawDepths = f['rawDepths']
        data_labels = f['labels']
        data_names = f['names']
        data_namesToIds = f['namesToIds']
        
        data_depth = data_depth[:]
        data_images = data_images[:]
        data_rawDepths = data_rawDepths[:]
        with open('input_data.npy','wb') as f:
            np.savez(f,x = data_depth,y = data_images,z = data_rawDepths)
        
    
    npzfile = np.load('input_data.npy')
    data_depth = npzfile['x']
    data_images = npzfile['y']
    data_rawDepths = npzfile['z']

    
    return data_depth,data_images,data_rawDepths
    
if __name__ == "__main__":
    data_depth,data_images,data_rawDepths = import_data()

        
    print(type(data_depth))
    print(type(data_images))
    print(type(data_rawDepths))
    print(data_images.shape)
    print(data_depth.shape)
    print(data_rawDepths.shape)
    
