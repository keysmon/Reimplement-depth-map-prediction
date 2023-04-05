import h5py
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# accelData – Nx4 matrix of accelerometer values indicated when each frame was taken. The columns contain the roll, yaw, pitch and tilt angle of the device.
# depths – HxWxN matrix of depth maps where H and W are the height and width, respectively and N is the number of images. The values of the depth elements are in meters.
# images – HxWx3xN matrix of RGB images where H and W are the height and width, respectively, and N is the number of images.
# labels – HxWxN matrix of label masks where H and W are the height and width, respectively and N is the number of images. The labels range from 1..C where C is the total number of classes. If a pixel’s label value is 0, then that pixel is ‘unlabeled’.
# names – Cx1 cell array of the english names of each class.
# namesToIds – map from english label names to IDs (with C key-value pairs)
# rawDepths – HxWxN matrix of depth maps where H and W are the height and width, respectively, and N is the number of images. These depth maps are the raw output from the kinect.
# scenes – Cx1 cell array of the name of the scene from which each image was taken.

def import_data():
    if not os.path.exists('input_data.npz'):
        f = h5py.File('nyu_depth_data_labeled.mat', 'r')
        
        data_depth = f['depths']
        data_images = f['images']
        data_rawDepths = f['rawDepths']
        data_labels = f['labels']
        data_names = f['names']
        data_namesToIds = f['namesToIds']
        
        # change data type from h5py to numpy
        #data_depth = data_depth[:]
        #data_images = data_images[:]
        
        # pre-process raw image data
        updated_data_images = []
        # change image shape from 3-h-w to h-w-3
        for image in data_images:
            r = image[0]
            g = image[1]
            b = image[2]
            dim = (int(r.shape[0]/2),int(r.shape[1]/2))
            r = cv2.pyrDown(r,dim)
            g = cv2.pyrDown(g,dim)
            b = cv2.pyrDown(b,dim)
            updated_image= cv2.merge([r,g,b])
            updated_data_images.append(updated_image)
        updated_data_images = np.array(updated_data_images)
        
        
        # pre-process data_depth
        updated_data_depth = []
        for depth in data_depth:
            dim = (int(depth.shape[0]/8),int(depth.shape[1]/8))
            updated_depth = cv2.pyrDown(depth,dim)
            updated_depth = cv2.pyrDown(updated_depth,dim)
            updated_depth = cv2.pyrDown(updated_depth,dim)

            updated_data_depth.append(updated_depth)
        #print(dim)
        #print(data_depth.shape)
        updated_data_depth = np.array(updated_data_depth)
        #print(updated_data_depth.shape)
            
        # save the data to a local file
        with open('input_data.npz','wb') as f:
            np.savez(f,x = updated_data_depth,y = updated_data_images)
        
    
    npzfile = np.load('input_data.npz')
    data_depth = npzfile['x']
    data_images = npzfile['y']
    
    
    return data_depth,data_images
   
def visualize_depth_map(samples):
    image, target = samples
    cmap = plt.cm.jet
    cmap.set_bad(color="black")


    fig, ax = plt.subplots(6, 2, figsize=(50, 50))
    for i in range(6):
        
        ax[i, 0].imshow((image[i].squeeze()))
        ax[i, 1].imshow((target[i].squeeze()), cmap=cmap)
    plt.show()


if __name__ == "__main__":
    data_depth,data_images, = import_data()
    #plt.imshow(data_images[0])
    #plt.show()
    visualize_depth_map((data_images,data_depth))
    print(type(data_depth))
    print(type(data_images))
    print(data_images.shape)
    print(data_depth.shape)
    
