import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


def load_dataset():
    '''
        Function to initialize variables and load them from .h5 file.

            [:] means we are getting the first to last elements -- to get all the rows 
            np.set_printoptions(threshold=np.inf)               -- to print all the data on terminal screen
            print(train_dataset["train_set_x"][:])              --  to print dataset we are reading

    hdf5 File structure
    
        train_catvnoncat
            - train_set_x (209 rows with 64 col.)
            - train_set_y (209 rows with 0/1)
            - list_classes (Cat? YES/NO)

    '''
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

##  Step 1. Loading the data into variables by calling our load_dataset function.
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes  =  load_dataset()

## Step 2. Viewing labelled dataset
index=11
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y_orig[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y_orig[:, index])].decode("utf-8") +  "' picture.")
plt.show()

