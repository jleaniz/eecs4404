import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def fit_LinRegr(X_train, y_train):
    # Get number of rows and columns from X_train
    rows = np.shape(X_train)[0]
    cols = np.shape(X_train)[1]
    # Create a vector of ones with `rows` number of rows
    ones = np.ones((rows,1))
    # Create a vector of zeroes with `cols`+1 columns
    w = np.zeros(cols + 1)
    # Add a column of ones to the original X_train matrix
    X_train = np.hstack((ones, X_train))

def mse(X_train,y_train,w):
    """This function returns the Mean Squared Error of the linear
    plane defined by vector w.
    """
    avgError = 0
    return avgError

def pred(X_train,w):
    """This function returns the predicted value for the specified data point of
    X_train.

    Returns: a real number representing the predicted value
    """
    

def test_SciKit(X_train, X_test, Y_train, Y_test):
    """This function uses SciKit Learn Linear Regression to train a model
    and calculate the Mean Squared Error.

    Returns: the MSE obtained by calling mean_squared_error.
    """
    sklearn_regressor = linear_model.LinearRegression()
    sklearn_regressor.fit(X_train, Y_train)
    pred_vals = sklearn_regressor.predict(X_test)
    error = mean_squared_error(Y_test, pred_vals)
    return error

def testFn_Part2():
    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2)
    
    #Testing Part 2a
    w=fit_LinRegr(X_train, y_train)
    e=mse(X_test,y_test,w)
    
    #Testing Part 2b
    scikit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Mean squared error from Part 2a is ", e)
    print("Mean squared error from Part 2b is ", scikit)

testFn_Part2()