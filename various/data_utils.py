import h5py

def get_cityscapes_data(input_dir='/mnt/localdata1/abailoni/HDF5/'):
    F_train = h5py.File(input_dir+"cityscapes_train_uint8.hdf5", "r")
    F_val = h5py.File(input_dir+"cityscapes_val_uint8.hdf5", "r")
    return F_train["train_x_dset"], F_train["train_y_dset"], F_val["val_x_dset"], F_val["val_y_dset"]

# def get_cityscapes_data_ymod(input_dir='/mnt/localdata1/abailoni/HDF5/'):
#     # input_dir = '/mnt/data2/abailoni/HDF5/'
#     F2 = h5py.File(input_dir+"cityscapes_train_ymod.hdf5", "r")
#     return F2["train_y_dset"]


def loaddata(path):
    """
    Function to load in a TIFF or HDF5 volume from file in `path`.

    :type path: str
    :param path: Path to file (must end with .tiff or .h5).

    Author: nasimrahaman
    """
    if path.endswith(".tiff") or path.endswith(".tif"):
        try:
            from vigra.impex import readVolume
        except ImportError:
            raise ImportError("Vigra is needed to read/write TIFF volumes, but could not be imported.")

        volume = readVolume(path)
        return volume

    elif path.endswith(".h5"):
        try:
            from Antipasti.netdatautils import fromh5
        except ImportError:
            raise ImportError("h5py is needed to read/write HDF5 volumes, but could not be imported.")

        volume = fromh5(path)
        return volume

    else:
        raise NotImplementedError("Can't load: unsupported format. Supported formats are .tiff and .h5")



# Save data
def savedata(data, path):
    """
    Saves volume as a .tiff or .h5 file in path.

    :type data: numpy.ndarray
    :param data: Volume to be saved.

    :type path: str
    :param path: Path to the file where the volume is to be saved. Must end with .tiff or .h5.

    Author: nasimrahaman
    """
    if path.endswith(".tiff") or path.endswith('.tif'):
        try:
            from vigra.impex import writeVolume
        except ImportError:
            raise ImportError("Vigra is needed to read/write TIFF volumes, but could not be imported.")

        writeVolume(data, path, '', dtype='UINT8')

    elif path.endswith(".h5"):
        try:
            from vigra.impex import writeHDF5
            vigra_available = True
        except ImportError:
            vigra_available = False
            import h5py

        if vigra_available:
            writeHDF5(data, path, "/data")
        else:
            with h5py.File(path, mode='w') as hf:
                hf.create_dataset(name='data', data=data)

    else:
        raise NotImplementedError("Can't save: unsupported format. Supported formats are .tiff and .h5")
