import numpy as np
from PIL import Image
import os
# import matplotlib.pyplot as plt

import h5py



# CROP_SIZE = 1024,2048
# MAX_CROP_SIZE = 1024,2048
CROP_SIZE = 256, 512
MAX_CROP_SIZE = 512, 512
IMAGE_SIZE = [1024,2048]
#PATCHES = 10


path = '/mnt/localdata01/abailoni/cityscape/leftImg8bit/val/'

file_list = [os.path.join(dirpath, f)
   for dirpath, dirnames, files in os.walk(path)
   for f in files if f.endswith('.png')]

inputs_val_x = sorted(file_list)

print(np.shape(inputs_val_x))

path = '/mnt/localdata01/abailoni/cityscape/gtFine/val/'

file_list = [os.path.join(dirpath, f)
   for dirpath, dirnames, files in os.walk(path)
   for f in files if f.endswith('gtFine_labelTrainIds.png')]

inputs_val_y = sorted(file_list)

print(np.shape(inputs_val_y))


output_dir = '/mnt/localdata01/abailoni/HDF5/'
F = h5py.File(output_dir+"cityscapes_val_uint8.hdf5", "w")

val_x = F.create_dataset("val_x_dset", (np.shape(inputs_val_x)[0],3,CROP_SIZE[0], CROP_SIZE[1]),
dtype='uint8')
val_y = F.create_dataset("val_y_dset", (np.shape(inputs_val_y)[0],CROP_SIZE[0], CROP_SIZE[1]), dtype='uint8')

# num_channels = 19
# NUM_CLASSES = 19

# Plot some images:
# from mod_nolearn.visualize import plot_images
from matplotlib import image
from matplotlib.pylab import imshow
import matplotlib.pyplot as plt

for i in range(len(inputs_val_x)):
   if i%10==0:
      print i
   try:
      im = Image.open(inputs_val_x[i])
      im.thumbnail(MAX_CROP_SIZE)
      target = Image.open(inputs_val_y[i])
      target.thumbnail(MAX_CROP_SIZE, Image.ANTIALIAS)
   except IOError:
      print "cannot create thumbnail for '%s'"

   img = image.pil_to_array(im)

   img = np.transpose(img,(2,0,1))

   # mean_train = np.reshape([[[ 73.15835921 , 82.90891754, 72.39239876]]],(3,1,1))
   # var_train = np.reshape([[[ 2237.79144756,  2326.24575092, 2256.68620499]]],(3,1,1))

   # img = np.subtract(img,mean_train)
   # img = np.divide(img ,np.sqrt(var_train))

   # img = img / 255.
   val_x[i,:,:,:] = img

   target = np.array(target)
   val_y[i,:,:] = target

   # if i==10:
   #    print val_x[i].shape
   #    fig = plot_images(val_x[:1])
   #    fig.savefig("__images__.pdf")
   #    break

F.close()
