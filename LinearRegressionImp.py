import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def fit_LinRegr(X_train, y_train):
    """This function computes the parameters of a vector w that best fits
    the training dataset
    """
    # Get number of rows and columns from X_train
    rows = np.shape(X_train)[0]
    cols = np.shape(X_train)[1]
    # Create a vector of ones with `rows` number of rows
    ones = np.ones((rows,1))
    # Create a vector of zeroes with `cols`+1 columns
    w = np.zeros(cols + 1)
    # Add a column of ones to the original X_train matrix
    X_train = np.hstack((ones, X_train))

    # Calculate X_train transpose, X_train transpose * X_train and
    # X_train transpose * y
    X_train_t = np.transpose(X_train)
    X_train_t_x = np.dot(X_train_t, X_train)
    X_train_t_y = np.dot(X_train_t, y_train)

    # Calculate the vector of coefficients w
    w = np.linalg.solve(X_train_t_x, X_train_t_y)

    return w

def mse(X_train,y_train,w):
    """This function returns the Mean Squared Error of the linear
    plane defined by vector w.
    """
    sum_error = 0.0
    rows = np.shape(X_train)[0]
    ones = np.ones((rows,1))
    X_train = np.hstack((ones, X_train))

    total_points = np.shape(X_train)[0]
    for ith_vector in range(total_points):
        pred_val = pred(X_train[ith_vector], w)
        #print("predicted: ", pred_val, "actual: ", y_train[ith_vector])
        pred_error = pred_val - y_train[ith_vector]
        sum_error += (pred_error ** 2)
    mse_value = sum_error / float(total_points)
    return mse_value

def pred(X_train,w):
    """This function returns the predicted value for the specified data point of
    X_train.

    Returns: a real number representing the predicted value
    """
    val = np.dot(X_train,w)
    return val

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

"""
Q: How close is the performance of your implementation in comparison to the existing modules in the
scikit-learn library?
A: The performance of our implementation is very close to the performance from scikit-learn
as seen in the sample runs below:

~ » /Users/juan.leaniz/anaconda3/bin/python /Users/juan.leaniz/Downloads/4404/AssignmentFiles/TestFiles/LinearRegressionImp.py
Mean squared error from Part 2a is  2710.895551989639
Mean squared error from Part 2b is  2710.8955519896394
(base) ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
~ » /Users/juan.leaniz/anaconda3/bin/python /Users/juan.leaniz/Downloads/4404/AssignmentFiles/TestFiles/LinearRegressionImp.py
Mean squared error from Part 2a is  2561.345836326977
Mean squared error from Part 2b is  2561.345836326976
(base) ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
~ » /Users/juan.leaniz/anaconda3/bin/python /Users/juan.leaniz/Downloads/4404/AssignmentFiles/TestFiles/LinearRegressionImp.py
Mean squared error from Part 2a is  2560.068526716539
Mean squared error from Part 2b is  2560.068526716539
(base) ---------------------------------------------------
"""
