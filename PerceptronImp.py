import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix 

def fit_perceptron(X_train, y_train):
    """ This fucntion trains a Perceptron.

    X_train: matrix of input features.
    y_train: vector of label values for each element.
    Return: A vector w containing the coefficients of the line computed
    by the Pocket algorithm.
    """
    max_iter = 1000
    learning_rate = 1
    # Get number of rows and columns from X_train
    rows = np.shape(X_train)[0]
    cols = np.shape(X_train)[1]
    # Create a vector of ones with `rows` number of rows
    ones = np.ones((rows,1))
    # Create a vector of zeroes with `cols`+1 columns
    w = np.zeros(cols + 1)
    # Add a column of ones to the original X_train matrix
    X_train = np.hstack((ones, X_train))

    for t in range(max_iter):
        for i, x in enumerate(X_train):
            #print("X_train i: %s  y_train i: %s  w: %s" % (X_train[i],y_train[i],w))
            val = np.dot(X_train[i], w) * y_train[i]
            if val <= 0:
                avgError = errorPer(X_train,y_train,w)
                w = w + learning_rate * X_train[i] * y_train[i]

    return w

def errorPer(X_train,y_train,w):
    """The output of this function is the average number of points that are
    misclassied by the plane dened by w.

    X_train: defined with the additional column of ones.
    y_train: defined in a manner similar to y_train in fit_perceptron.
    w: represents the coecients of a linear plane and is of d + 1 dimensions.
    """
    misclassified_count = 0
    total_points = np.shape(X_train)[0]
    avgError = 0
    for i, x in enumerate(X_train):
        pred_val = pred(X_train[i], w)
        if pred_val != y_train[i]:
            misclassified_count += 1
    avgError = misclassified_count / total_points
    return avgError

def confMatrix(X_train,y_train,w):
    """This function returns a two-by-two matrix composed of integer values
    that represent the how well the Perceptron performed in classifying 
    the input values.
    """
    # Need to add a column of ones to X_train given that
    # w is of dimensions d + 1 as specified in the output of
    # fit_perceptron()
    rows = np.shape(X_train)[0]
    ones = np.ones((rows,1))
    X_train = np.hstack((ones, X_train))
    preds = []
    # Create an empty 2x2 matrix of integer type
    conf_matrix = np.zeros((2,2), dtype=int)

    # Calculate the predicted values and fill in the confusion matrix
    for i, x in enumerate(X_train):
        pred_val = pred(X_train[i], w)
        preds.append(pred_val)
        # True Negatives
        if pred_val == y_train[i] and y_train[i] == -1:
            conf_matrix[0,0] += 1
        # False Negatives
        elif pred_val != y_train[i] and y_train[i] == 1:
            conf_matrix[1,0] += 1
        # True Positives
        elif pred_val == y_train[i] and y_train[i] == 1:
            conf_matrix[1,1] += 1
        # False Positives
        elif pred_val != y_train[i] and y_train[i] == -1:
            conf_matrix[0,1] += 1
    
    #print("predicted values: ", preds)
    return conf_matrix
    
def pred(X_train,w):
    """This function calculates the class label (1 or -1) for
    an X_i feature vector.

    Returns: 1 if dot product is strictly positive, -1 otherwise.
    """
    val = X_train.dot(w)
    if val > 0:
        return 1
    else:
        return -1

def test_SciKit(X_train, X_test, Y_train, Y_test):
    """This function returns the confusion matrix of the linear model trained
    using the Perceptron algorithm. This function uses the SciKit-Learn library.
    """
    ppn = Perceptron(max_iter=1000, eta0=1)
    ppn.fit(X_train, Y_train)
    y_pred = ppn.predict(X_test)
    #print("scikit-learn results")
    #print("Y_test: ", Y_test)
    #print("y_pred: ", y_pred)
    return confusion_matrix(Y_test, y_pred)

def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2)
    
    for i in range(80):
        if y_train[i]==1:
            y_train[i]=-1
        else:
            y_train[i]=1
    for j in range(20):
        if y_test[j]==1:
            y_test[j]=-1
        else:
            y_test[j]=1
            
    #Testing Part 1a
    w=fit_perceptron(X_train,y_train)
    cM=confMatrix(X_test,y_test,w)
    
    #Testing Part 1b
    sciKit = test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)
    
test_Part1()

"""
Q: How close is the performance of your implementation in comparison to the existing modules in the scikit-learn
library? Place this comment at the end of the code le.

A: In the tests performed, our implementation produces results that are very similar to scikit-learn but not quite
the same. In various runs of the program, the results varied slightly but not significantly.

The text below illustrates three different runs of the PerceptronImp.py program:

~ » /Users/juan.leaniz/anaconda3/bin/python /Users/juan.leaniz/Downloads/4404/AssignmentFiles/TestFiles/PerceptronImp.py
Confusion Matrix is from Part 1a is:  [[10  0]
 [ 2  8]]
Confusion Matrix from Part 1b is: [[10  0]
 [ 0 10]]
~ » /Users/juan.leaniz/anaconda3/bin/python /Users/juan.leaniz/Downloads/4404/AssignmentFiles/TestFiles/PerceptronImp.py
Confusion Matrix is from Part 1a is:  [[8 2]
 [1 9]]
Confusion Matrix from Part 1b is: [[ 8  2]
 [ 0 10]]
~ » /Users/juan.leaniz/anaconda3/bin/python /Users/juan.leaniz/Downloads/4404/AssignmentFiles/TestFiles/PerceptronImp.py
Confusion Matrix is from Part 1a is:  [[10  1]
 [ 0  9]]
Confusion Matrix from Part 1b is: [[7 4]
 [0 9]]
"""
