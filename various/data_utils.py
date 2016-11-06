import h5py

def get_cityscapes_data():
    # input_dir = '/mnt/data2/abailoni/HDF5/'
    input_dir = '/mnt/localdata1/abailoni/HDF5/'
    F_train = h5py.File(input_dir+"cityscapes_train_uint8.hdf5", "r")
    F_val = h5py.File(input_dir+"cityscapes_val_uint8.hdf5", "r")
    #


    return F_train["train_x_dset"], F_train["train_y_dset"], F_val["val_x_dset"], F_val["val_y_dset"]

def get_cityscapes_data_ymod():
    # input_dir = '/mnt/data2/abailoni/HDF5/'
    input_dir = '/mnt/localdata1/abailoni/HDF5/'
    F2 = h5py.File(input_dir+"cityscapes_train_ymod.hdf5", "r")
    return F2["train_y_dset"]
