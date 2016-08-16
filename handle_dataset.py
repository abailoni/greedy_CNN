import numpy as np
from PIL import Image
import os
# import matplotlib.pyplot as plt

import h5py



CROP_SIZE = 256, 512
MAX_CROP_SIZE = 512, 512
IMAGE_SIZE = [1024,2048]
#PATCHES = 10


path = '/export/home/abailoni/cityscapes/leftImg8bit/train/'

file_list = [os.path.join(dirpath, f)
   for dirpath, dirnames, files in os.walk(path)
   for f in files if f.endswith('.png')]

inputs_train_x = sorted(file_list)

print(np.shape(inputs_train_x))

path = '/export/home/abailoni/cityscapes/gtFine/train/'

file_list = [os.path.join(dirpath, f)
   for dirpath, dirnames, files in os.walk(path)
   for f in files if f.endswith('gtFine_labelTrainIds.png')]

inputs_train_y = sorted(file_list)

print(np.shape(inputs_train_y))


output_dir = '/mnt/localdata01/abailoni/HDF5/'
F = h5py.File(output_dir+"cityscapes_train.hdf5", "w")

train_x = F.create_dataset("train_x_dset", (2975,3,CROP_SIZE[0], CROP_SIZE[1]),
dtype='float32')
train_y = F.create_dataset("train_y_dset", (2975,CROP_SIZE[0], CROP_SIZE[1]), dtype='uint8')

# num_channels = 19
# NUM_CLASSES = 19


for i in range(len(inputs_train_x)):
   print i
   try:
      im = Image.open(inputs_train_x[i])
      im.thumbnail(MAX_CROP_SIZE, Image.ANTIALIAS)
      target = Image.open(inputs_train_y[i])
      target.thumbnail(MAX_CROP_SIZE, Image.ANTIALIAS)
   except IOError:
      print "cannot create thumbnail for '%s'"

   img = np.array(im)
   img = np.transpose(img,(2,0,1))

   mean_train = np.reshape([[[ 73.15835921 , 82.90891754 ,
72.39239876]]],(3,1,1))
   var_train = np.reshape([[[ 2237.79144756,  2326.24575092,
2256.68620499]]],(3,1,1))

   img = np.subtract(img,mean_train)
   img = np.divide(img ,np.sqrt(var_train))

   target = np.array(target)

   train_x[i,:,:,:] = img
   train_y[i,:,:] = target

F.close()
