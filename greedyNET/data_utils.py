import h5py

def get_cityscapes_data():
    # input_dir = '/mnt/data2/abailoni/HDF5/'
    input_dir = '/mnt/localdata01/abailoni/HDF5/'
    F = h5py.File(input_dir+"cityscapes_train_uint8.hdf5", "r")
    F2 = h5py.File(input_dir+"cityscapes_train_ymod.hdf5", "r")


    return F["train_x_dset"], F["train_y_dset"], F2["train_y_dset"]
