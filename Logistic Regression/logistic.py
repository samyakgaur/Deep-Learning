import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import cv2 



img = cv2.imread("samyakgaur.jpg")   # reads an image in the BGR format
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR -> RGB
img=img.reshape(img.shape[0]*img.shape[1]*3,1)
print(img.shape)

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
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes  =  load_dataset()

## Step 2. Viewing labelled dataset
'''
    - Uncomment the following code to view image.
        index=11
        plt.imshow(train_set_x_orig[index])
        print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
        plt.show()
'''

## Step 3. Reshaping the image array to be a numpy array of (numpy_x * numpy_x *3 , 1 )
''' each column represents one image.
    shape of the following variables:
        -train_x_orig        :  (209,64,64,3) 209 images/rows with 3(rgb), 64x64 2d matrix values
        -train_set_y_orig    :  (1,209)       1d array with 209 elements  
        -train_set_x_flatten :  (12288,209)
        -test_x_orig         :  (50,64,64,3)   
        -test_set_y_orig     :  (1,50)   
        -test_set_x_flatten  :  (12288,50)
'''
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten  = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
#Standardizing data by subtracting the mean from the np.array
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255


## Step 4: Inititalize the parameters w,b with 0
def initialize_with_zero(dimension):
    '''
    initialized w i.e the theta to a matrix with `dimension` rows and 1 column and b=0
    '''
    w  =  np.zeros((dimension,1))
    b  =  0
    return w,b

## Step 5: Make a function of calculate sigmoid 
def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

## Step 6: Function for the propagation of w,b i.e backward & forward propagation
def propagate(w,b,X,Y):
    '''
        w,b = parameters
        X,Y = training examples

        returns:
        cost,dw,db
    '''
    m=X.shape[1]  #Number of examples in training set 

    #Forward Propagation
    A  =  sigmoid(np.dot(w.T,X)+b)
    cost = -1./m* np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)) #calculation the avg devaition from the actual value
    #Backward Propagation
    dw = 1./m*np.dot(X, (A-Y).T)
    db = 1./m*np.sum(A-Y)

    #returning

    grads = {"dw":dw,
             "db":db   
                }
    return grads

## Step 7: Function to optimize w,b by running gradient decent algo
def optimize(w, b, X, Y, num_iterations, learning_rate):
    '''
        returns
        parameters - w,b
        grads      - gradients of w,b
        costs       - cost
    '''
    for i in range(num_iterations):
        # Run gradient decend algo
        grads = propagate(w, b, X, Y)
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        # Update values of parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db
    
    #return 
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads

# Step 8: Predict the value using w,b found
def predict(w,b,X):
    # number of elements
    m = X.shape[1]
    # array containing predictions
    Y_prediction = np.zeros((1,m))
    '''
        w=[
            1
            2
            3
            .
            .
            .   
            m
              ]
    '''
    w = w.reshape(X.shape[0], 1)
    
    # bring the value of y b/w 0/1 using w and b calculates
    A =   sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
             Y_prediction[0, i] = 0
    
    return Y_prediction

# Main Function
def model(X_train,Y_train,X_test,Y_test,num_iterations=2000,learning_rate=0.5):
    '''
    Builds the logistic regression model by calling the function you've implemented previously
    
        Arguments:
        X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
        Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
        X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
        Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_cost -- Set to true to print the cost every 100 iterations
        
        Returns:
        d -- dictionary containing information about the model.
    '''

    w,b  =  initialize_with_zero(X_train.shape[0])
    
    #gradient decent 
    parameters, grads = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test  = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    #Testing on my image
    prediction =  sigmoid(np.dot(w.T, img) + b)
    print("My Picture : "+str((prediction[0]>0.5) and "is a cat picture" or "not a cat picture"))


    d = {
        "Y_prediction_test"  : Y_prediction_test,  
        "Y_prediction_train" : Y_prediction_train, 
        "w" : w, 
        "b" : b,
        "learning_rate" : learning_rate,
        "num_iterations": num_iterations
        }

    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005)
index = 46
num_px = test_set_x_orig.shape[1]
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
# print(d["Y_prediction_test"][0,46])
# print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[d["Y_prediction_test"][0,index]].decode("utf-8") +  "\" picture.")
# plt.show()



