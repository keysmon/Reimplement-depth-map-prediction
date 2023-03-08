import h5py
import pandas as pd


# accelData – Nx4 matrix of accelerometer values indicated when each frame was taken. The columns contain the roll, yaw, pitch and tilt angle of the device.
# depths – HxWxN matrix of depth maps where H and W are the height and width, respectively and N is the number of images. The values of the depth elements are in meters.
# images – HxWx3xN matrix of RGB images where H and W are the height and width, respectively, and N is the number of images.
# labels – HxWxN matrix of label masks where H and W are the height and width, respectively and N is the number of images. The labels range from 1..C where C is the total number of classes. If a pixel’s label value is 0, then that pixel is ‘unlabeled’.
# names – Cx1 cell array of the english names of each class.
# namesToIds – map from english label names to IDs (with C key-value pairs)
# rawDepths – HxWxN matrix of depth maps where H and W are the height and width, respectively, and N is the number of images. These depth maps are the raw output from the kinect.
# scenes – Cx1 cell array of the name of the scene from which each image was taken.
def import_data():
    f = h5py.File('nyu_depth_data_labeled.mat', 'r')
    data_depth = f['depths']
    data_images = f['images']
    data_rawDepths = f['rawDepths']
    data_labels = f['labels']
    data_names = f['names']
    data_namesToIds = f['namesToIds']
    return data_depth,data_images,data_rawDepths,data_labels,data_names,data_namesToIds

import_data()
