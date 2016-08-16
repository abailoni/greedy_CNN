import numpy as np
from PIL import Image
import os
# import matplotlib.pyplot as plt

import h5py



CROP_SIZE = 256, 512
MAX_CROP_SIZE = 512, 512
IMAGE_SIZE = [1024,2048]
NUM_CLASSES = 2
#PATCHES = 10

path = '/export/home/abailoni/cityscapes/gtFine/train/'

file_list = [os.path.join(dirpath, f)
   for dirpath, dirnames, files in os.walk(path)
   for f in files if f.endswith('gtFine_labelTrainIds.png')]

inputs_train_y = sorted(file_list)

print(np.shape(inputs_train_y))


output_dir = '/mnt/localdata01/abailoni/HDF5/'
F = h5py.File(output_dir+"cityscapes_train_ymod.hdf5", "w")

train_y = F.create_dataset("train_y_dset", (2975,NUM_CLASSES, CROP_SIZE[0], CROP_SIZE[1]), dtype='int8')

# num_channels = 19


for i in range(len(inputs_train_y)):
   print i
   try:
      target = Image.open(inputs_train_y[i])
      target.thumbnail(MAX_CROP_SIZE, Image.ANTIALIAS)
   except IOError:
      print "cannot create thumbnail for '%s'"

   target = np.array(target)

   dim_mod = NUM_CLASSES, target.shape[0], target.shape[1]
   target_mod = np.zeros(dim_mod, dtype=np.int8)
   idxs = np.indices(target.shape)
   idxs_flat = [idxs[0].flatten(), idxs[1].flatten()]
   target_mod[target[idxs_flat[0],idxs_flat[1]], idxs_flat[0], idxs_flat[1]] = 1
   train_y[i,:,:,:] = target_mod

F.close()
