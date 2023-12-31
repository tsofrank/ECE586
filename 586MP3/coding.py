''' working progress'''
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import rand, randn
from scipy.optimize import linprog

import numpy as np
#from numba import njit
from scipy.linalg import inv, svd
from tqdm.notebook import tqdm

'''to import'''
import numpy.linalg as LA
import math
from sklearn.model_selection import train_test_split
import copy
import sys
### Helper functions
# Function to remove outliers before plotting histogram
def remove_outlier(x, thresh=3.5):
    """
    returns points that are not outliers to make histogram prettier
    reference: https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting/11886564
    Arguments:
        x {numpy.ndarray} -- 1d-array, points to be filtered
        thresh {float} -- the modified z-score to use as a threshold. Observations with
                          a modified z-score (based on the median absolute deviation) greater
                          than this value will be classified as outliers.
    Returns:
        x_filtered {numpy.ndarray} -- 1d-array, filtered points after dropping outlier
    """
    if len(x.shape) == 1: x = x[:,None]
    median = np.median(x, axis=0)
    diff = np.sqrt(((x - median)**2).sum(axis=-1))
    modified_z_score = 0.6745 * diff / np.median(diff)
    x_filtered = x[modified_z_score <= thresh]
    x_filtered = x_filtered.flatten()
    return x_filtered

## End of helper functions

# Compute null space
def null_space(A, rcond=None):
    """
    Compute null spavce of matrix XProjection on half space defined by {v| <v,w> = c}
    Arguments:
        A {numpy.ndarray} -- matrix whose null space is desired
        rcond {float} -- intercept
    Returns:
        Q {numpy.ndarray} -- matrix whose (rows?) span null space of A
    """
    u, s, vh = svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q

### End Helper Functions

# Exercise 1: Alternating projection for subspaces
def altproj(A, B, v0, n):
    """
    Arguments:
        A {numpy.ndarray} -- matrix whose columns form basis for subspace U
        B {numpy.ndarray} -- matrix whose columns form baiss for subspace W
        v0 {numpy.ndarray} -- initialization vector
        n {int} -- number of sweeps for alternating projection
    Returns:
        v {numpy.ndarray} -- the output after 2n steps of alternating projection
        err {numpy.ndarray} -- the error after each full pass
    """
    
    ### Add code here
    UintW = np.hstack([A, B]) @ null_space(np.hstack([A, -B])) # The intersection of U andW
    
    # projection matrix onto range of A and range of B and range of UintW
    P_A = A @ LA.inv(A.T @ A) @ A.T # orhtogonal projection onto U = range(A)
    P_B = B @ LA.inv(B.T @ B) @ B.T # orthogonal projection onto W = range(B)
    P_UintW = UintW @ LA.inv(UintW.T @ UintW) @ UintW.T

    res = P_UintW.dot(v0) # projection of v0 onto U int W

    err = []
    v = v0 # start with v0
    for i in range(n):
        # first project onto U then project onto W
        v = P_A.dot(v)
        v = P_B.dot(v)
        g_2k = LA.norm(v - res, np.inf)
        err.append(g_2k)

    return v, err

# Exercise 2: Kaczmarz algorithm for solving linear systems
#@njit
def kaczmarz(A, b, I):
    """
    Arguments:
        A {numpy.ndarray} -- matrix defines the LHS of linear equation
        b {numpy.ndarray} -- vector defines the RHS of linear equation
        I {int} -- number of full passes through the Kaczmarz algorithm
    Returns:
        X {numpy.ndarray} -- the output of all I full passes
        err {numpy.ndarray} -- the error after each full pass
    """
    
    ### Add code here
    m,n = A.shape
    v = np.zeros(n) # v0 = 0
    X = []
    err = []
    for i in range(I*m):
        sigma = i % m
        a = A[sigma]
        num = np.dot(v,a) - b[sigma]
        denom = np.dot(a,a)
        v = v - num/denom * a
        if i % m == (m-1) and i > 0:
            gk = LA.norm(A.dot(v)-b, np.inf)
            err.append(gk)
            X.append(v)

    X = np.array(X).T
    err = np.array(err)
    return X, err

# Exercise 4: Alternating projection to satisfy linear inequalities
#@njit
def lp_altproj(A, b, I, s=1, d=0):
    """
    Find a feasible solution for A v >= b using alternating projection
    starting from v0 = 0
    Arguments:
        A {numpy.ndarray} -- matrix defines the LHS of linear equation
        b {numpy.ndarray} -- vector defines the RHS of linear equation
        I {int} -- number of full passes through the alternating projection
        s {numpy.float} -- step size of projection (defaults to 1)
        d {numpy.float} -- coordinate lower bound for all elements of v (defaults to 0)
    Returns:
        v {numpy.ndarray} -- the output after I full passes
        err {numpy.ndarray} -- the error after each full pass
    """
    
    # Add code here
    m,n = A.shape
    v = np.zeros(n)
    X = []
    err = []
    for i in range(m*I):
        sigma = i % m
        a = A[sigma]
        if b[sigma] - np.dot(a,v) <= 0: # if already in the half space
            dir = v
        else: # project onto the hyperplane a^T x = b
            dir = v - (np.dot(v,a)-b[sigma])/np.dot(a,a) * a
        v = (1-s)*v + s*dir

        # check if lower bound for v >= d is satisfied
        if d != -np.inf: # when there is a coordinate bound for variables
            for j, val in enumerate(v):
                if val < d:
                    v[j] = d

        if i % m == (m-1) and i > 0: # after each round with constraints Av >= b
            k = math.ceil(i/m) # counter
            if k % 10 == 1: print("k = {}".format(k)) # check to see code is running
            X.append(v)
            gk = np.max([np.max(b - A.dot(v)),0]) # if in the convex set, then gk = 0
            err.append(gk)
    
    return v, err

# General function to extract samples with given labels and randomly split into test and training sets for Exercise 2.3
def extract_and_split(df, d, ts):
    """
    extract the samples with given labels and randomly separate the samples into training and testing groups, extend each vector to length 785 by appending a −1
    Arguments:
        df {dataframe} -- the dataframe of MNIST dataset
        d {int} -- digit needs to be extracted, can be 0, 1, ..., 9
        test_size {float} -- the fraction of testing set, default value is 0.5
    Returns:
        X_tr {numpy.ndarray} -- training set features, a matrix with 785 columns
                                each row corresponds the feature of a sample
        y_tr {numpy.ndarray} -- training set labels, 1d-array
                                each element corresponds the label of a sample
        X_te {numpy.ndarray} -- testing set features, a matrix with 785 columns 
                                each row corresponds the feature of a sample
        y_te {numpy.ndarray} -- testing set labels, 1d-array
                                each element corresponds the label of a sample
    """
    df1 = df.loc[df['label'] == d] # obtain only vectors corresponding to label
    X = df1.loc[:,'feature']
    Y = df1.loc[:,'label']

    X = X.apply(lambda x: np.append(x,-1))

    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = ts, random_state=42,shuffle=True)

    X_tr = np.stack(X_train.values)
    X_te = np.stack(X_test.values)
    y_tr = np.stack(y_train.values)
    y_te = np.stack(y_test.values)
    return X_tr, X_te, y_tr, y_te  

# General function to train and test pairwise classifier for MNIST digits for Exercise 3.2
def mnist_pairwise_altproj(df, a, b, solver, test_size = 0.5, verbose=False):
    """
    Pairwise experiment for applying least-square to classify digit a and digit b
    Arguments:
        df {dataframe} -- the dataframe of MNIST dataset
        a, b {int} -- digits to be classified
        test_size {float} -- the fraction of testing set, default value is 0.5
        verbose {bool} -- whether to print and plot results
        gd {bool} -- whether to use gradient descent to solve LS        
    Returns:
        res {numpy.ndarray} -- numpy.array([training error, testing error])
    """
    # Find all samples labeled with digit a and split into train/test sAbets
    Xa_tr, Xa_te, ya_tr, ya_te = extract_and_split(df, a, test_size)
    ya_tr = -np.ones(len(ya_tr))
    ya_te = -np.ones(len(ya_te))

    # Find all samples labeled with digit b and split into train/test sets
    Xb_tr, Xb_te, yb_tr, yb_te = extract_and_split(df,b, test_size)
    yb_tr = np.ones(len(yb_tr))
    yb_te = np.ones(len(yb_te))

    # Construct the full training set
    X_tr = np.concatenate([-Xa_tr, Xb_tr]) # this is our A_tilde
    c = np.concatenate([-ya_tr, yb_tr]) # this is our c for Ax >= c
    y_tr = np.concatenate([ya_tr, yb_tr])
    
    # Construct the full testing set
    y_te = np.concatenate([ya_te, yb_te])

    # shuffle training set
    Ab = np.vstack([X_tr.T, y_tr])
    Ab = np.transpose(Ab)
    np.random.shuffle(Ab)
    X_tr = Ab[:,:-1]
    y_tr = Ab[:,-1]
    X_tr_tilde = copy.deepcopy(X_tr) # used for testing result 
    for j, val in enumerate(y_tr):
        if val < 0:
            X_tr_tilde[j] *= -1

    z_hat, err = solver(X_tr, c)
    # print(np.all(X_tr @ z_hat - c > 0))

    # plt.figure(figsize=(8, 6))
    # plt.semilogy(np.arange(1, 100 + 1), err)
    # plt.show()

    # Compute estimate and classification error for training set
    y_hat_tr = X_tr_tilde.dot(z_hat)
    # convert y_hat entries to -1 and 1
    y_hat_tr = np.where(y_hat_tr >1, 1, y_hat_tr)
    y_hat_tr = np.where(y_hat_tr <-1, -1, y_hat_tr)

    err_tr = np.count_nonzero(y_tr - y_hat_tr) / len(y_tr)

    # Compute estimate and classification error for training set
    y_hat_te = np.concatenate([Xa_te, Xb_te]).dot(z_hat)
    # convert y_hat entries to -1 and 1
    y_hat_te = np.where(y_hat_te >1, 1, y_hat_te)
    y_hat_te = np.where(y_hat_te <-1, -1, y_hat_te)
    err_te = np.count_nonzero(y_te - y_hat_te) / len(y_te)
    
    '''Compare step sizes'''
    # z_hat1, err1 = lp_altproj(X_tr, c + 1e-6, 100, 0.1, -np.Inf)
    # # Compute estimate and classification error for training set
    # y_hat_tr1 = X_tr_tilde.dot(z_hat1)
    # # convert y_hat entries to -1 and 1
    # y_hat_tr1 = np.where(y_hat_tr1 >1, 1, y_hat_tr1)
    # y_hat_tr1 = np.where(y_hat_tr1 <-1, -1, y_hat_tr1)

    # err_tr1 = np.count_nonzero(y_tr - y_hat_tr1) / len(y_tr)
    
    # # Compute estimate and classification error for training set
    # y_hat_te1 = np.concatenate([Xa_te, Xb_te]).dot(z_hat1)
    # # convert y_hat entries to -1 and 1
    # y_hat_te1 = np.where(y_hat_te1 >1, 1, y_hat_te1)
    # y_hat_te1 = np.where(y_hat_te1 <-1, -1, y_hat_te1)
    # err_te1 = np.count_nonzero(y_te - y_hat_te1) / len(y_te)
    # print('training error = {0:.2f}%, testing error = {1:.2f}%'.format(100 * err_tr, 100 * err_te))
    # print('training error = {0:.2f}%, testing error = {1:.2f}%'.format(100 * err_tr1, 100 * err_te1))
    # # conclusion, same I rounds, smaller s results in worse performance.
    '''end of compare step sizes'''

    if verbose:
        print('Pairwise experiment, mapping {0} to -1, mapping {1} to 1'.format(a, b))
        print('training error = {0:.2f}%, testing error = {1:.2f}%'.format(100 * err_tr, 100 * err_te))
        
        # Compute confusion matrix
        cm = np.zeros((2, 2), dtype=np.int64)
        cm[0, 0] = ((y_te == -1) & (y_hat_te == -1)).sum()
        cm[0, 1] = ((y_te == -1) & (y_hat_te == 1)).sum()
        cm[1, 0] = ((y_te == 1) & (y_hat_te == -1)).sum()
        cm[1, 1] = ((y_te == 1) & (y_hat_te == 1)).sum()
        print('Confusion matrix:\n {0}'.format(cm))

        # Compute the histogram of the function output separately for each class 
        # Then plot the two histograms together
        ya_te_hat, yb_te_hat = Xa_te @ z_hat, Xb_te @ z_hat
        output = [remove_outlier(ya_te_hat), remove_outlier(yb_te_hat)]
        plt.figure(figsize=(8, 4))
        plt.hist(output, bins=50)
    
    res = np.array([err_tr, err_te])
    return z_hat, res
''' exercise 1'''
# A = np.array([[3, 1, 4, 1, 5], [2, 5, 11, 17, 23], [3, 7, 13, 19, 29]]).T
# B = np.array([[1, 2, 2, 2, 6], [1, 0, 1, 0, -3], [2.5, 6, 12, 18, 26]]).T
# v0 = np.array([1, 2, 3, 4, 5])
# n = 20
# v, err = altproj(A, B, v0, n)

# plt.figure(figsize=(8, 6))
# plt.semilogy(np.arange(1, n + 1), err)
# plt.show()

'''exercise 2'''
# A = np.array([[2, 5, 11, 17, 23], [3, 7, 13, 19, 29]])
# b = np.array([228, 227])
# I = 500

# X, err = kaczmarz(A, b, I)

# plt.figure(figsize=(8, 6))
# plt.semilogy(np.arange(1, I + 1), err)
# plt.show()

'''exercise 4'''
# c = np.array([3, -1, 2])
# A = np.array([[2, -1, 1], [1, 0, 2], [-7, 4, -6]])
# b = np.array([-1, 2, 1])

# res = linprog(c, A_ub=-A, b_ub=-b, bounds=[(0, None)] * c.size, method='interior-point')
# print(res)
# I = 2000
# # Do not forget constraint xi >= 0
# A1 = np.append(A,-c).reshape(4,3)
# b1 = np.append(b,0)

# x, err = lp_altproj(A1, b1, I, d=0) # input d= -np.inf if there is no lower bound for variable

# plt.figure(figsize=(8, 6))
# plt.semilogy(np.arange(1, I + 1), err)
# plt.show()

# x = np.round(x,10)
# print(np.all(A @ x - b >= 0), np.all(x >= 0))
'''exercise 4'''
# c = np.array([3, -1, 2])
# A = np.array([[2, -1, 1], [1, 0, 2], [-7, 4, -6]])
# b = np.array([-1, 2, 1])

# res = linprog(c, A_ub=-A, b_ub=-b, bounds=[(0, None)] * c.size, method='interior-point')
# print(res)

# I = 500
# # Do not forget constraint xi >= 0
# A1 = np.append(A,-c).reshape(4,3)
# b1 = np.append(b,0) 
# x, err = lp_altproj(A1, b1, I) # input d='None' if there is no lower bound for variable

# plt.figure(figsize=(8, 6))
# plt.semilogy(np.arange(1, I + 1), err)
# plt.show()
'''exercise 5'''
# np.random.seed(42)
# c = randn(1000)
# A = np.vstack([-np.ones((1, 1000)), randn(500, 1000)])
# b = np.concatenate([[-1000], A[1:] @ rand(1000)])

# I, ep = 1000, 1e-6
# # Do not forget constraint xi >= 0, and c^T x <= -1000
# A1 = np.append(A,-c).reshape(len(A)+1,len(A[0]))
# b1 = np.append(b,1000)
# x, err = lp_altproj(A1, b1 + ep, I, s = 1, d = ep)
# print(np.all(A @ x - b >= 0), np.all(x >= 0), c.reshape(1, -1) @ x)

# plt.figure(figsize=(8, 6))
# plt.semilogy(np.arange(1, I + 1), err)
# plt.show()
# res = linprog(c, A_ub=-A, b_ub=-b, bounds=[(0, None)] * 1000, method='interior-point')
# print(res.fun)

'''exercise 6'''
# Read MNIST csv file into dataframe
df = pd.read_csv('mnist_train.csv')
# append feature column by merging all pixel columns
df['feature'] = df.apply(lambda row: row.values[1:], axis=1)
# only keep feature and label column
df = df[['feature', 'label']]

solver = lambda A, b: lp_altproj(A, b + 1e-6, 100, 1, -np.Inf)
z_hat, res = mnist_pairwise_altproj(df, 2, 3, solver, verbose=True)

# debug break here
a = 1