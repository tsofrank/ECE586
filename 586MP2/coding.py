import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it
from scipy.sparse import coo_matrix
from numpy import linalg
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

## Helper functions you don't need to modify

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

## Coding Exercise Starts Here

# General function to compute Vandermonde matrix for Exercise 2.2
def create_vandermonde(x, m):
    """
    Arguments:
        x {numpy.ndarray} -- 1d-array of (x_1, x_2, ..., x_n)
        m {int} -- a non-negative integer, degree of polynomial fit
    Returns:
        A {numpy.ndarray} -- an n x (m+1) matrix where A_{ij} = x_i^{j-1}
    """
    # Add code to compute Vandermonde A
    n = len(x)
    A = np.zeros((n,m+1))
    for i in range(n):
        val = x[i]
        for j in range(m+1):
            A[i][j] = val**j
            # print("i: {0}, j: {1}".format(i, j))
    return A

# General function to solve linear least-squares via normal equations for Exercise 2.2
def solve_linear_LS(A, y):
    """
    Arguments:
        A {numpy.ndarray} -- an m x n matrix
        y {numpy.ndarray} -- a length-m vector
    Returns:
        z_hat {numpy.ndarray} -- length-n vector, the optimal solution for the given linear least-square problem
    """
    # Add code to compute least squares solution z_hat via linear algebra
    z_hat = np.linalg.solve(np.dot(A.T,A),A.T.dot(y)) # solve normal equation
    return z_hat

# General function to solve linear least-squares via via partial gradient descent for Exercise 2.2
def solve_linear_LS_gd(A, y, step, niter):
    """
    Arguments:
        A {numpy.ndarray} -- an m x n matrix
        y {numpy.ndarray} -- a length-m vector
        step -- a floating point number, step size
        niter -- a non-negative integer, number of updates
    Returns:
        z_hat {numpy.ndarray} -- length-n vector, the optimal solution for the given linear least-square problem
    """
    # Add code to approximate least squares solution z_hat via gradient descent
    N,M = A.shape
    z = np.zeros(M)
    mse = 0.18214285714285716
    mse2 = 1000
    T = 0 # counter
    r = step # step size gamma_t
    error = 1
    while T < niter and mse2 > 1.2 * mse:
        # update z
        index = T % N
        a = A[index]
        z = z + r*(y[index]-np.dot(a,z))*a
        
        # compute MSE
        f = A.dot(z)
        error = y - f
        mse2 = 1/len(y) * np.inner(error,error)

        T += 1
        
    z_hat = z
    return z_hat, mse2, T-1

# General function to extract samples with given labels and randomly split into test and training sets for Exercise 2.3
def extract_and_split(df, d, test_size=0.5):
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
    df1 = df.loc[df['label'] ==d] # obtain only vectors corresponding to label
    X = df1.loc[:,'feature']
    Y = df1.loc[:,'label']

    X = X.apply(lambda x: np.append(x,-1))

    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.5, random_state=42,shuffle=True)

    X_tr = np.stack(X_train.values)
    X_te = np.stack(X_test.values)
    y_tr = np.stack(y_train.values)
    y_te = np.stack(y_test.values)
    return X_tr, X_te, y_tr, y_te  


# General function to train and test pairwise classifier for MNIST digits for Exercise 3.2
def mnist_pairwise_LS(df, a, b, test_size=0.5, verbose=False, gd=False):
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
    # Find all samples labeled with digit a and split into train/test sets
    Xa_tr, Xa_te, ya_tr, ya_te = extract_and_split(df,d=a)
    ya_tr = -1*np.ones(len(ya_tr))
    ya_te = -1*np.ones(len(ya_te))

    # Find all samples labeled with digit b and split into train/test sets
    Xb_tr, Xb_te, yb_tr, yb_te = extract_and_split(df,d=b)
    yb_tr = np.ones(len(yb_tr))
    yb_te = np.ones(len(yb_te))

    # Construct the full training set
    X_tr = np.concatenate([Xa_tr, Xb_tr])
    y_tr = np.concatenate([ya_tr, yb_tr])
    
    # Construct the full testing set
    X_te = np.concatenate([Xa_te, Xb_te])
    y_te = np.concatenate([ya_te, yb_te])
    
    # Run least-square on training set
    z_hat = np.linalg.lstsq(X_tr,y_tr)[0]

    # Compute estimate and classification error for training set
    y_hat_tr = X_tr.dot(z_hat)
    # convert y_hat entries to -1 and 1
    y_hat_tr = np.where(y_hat_tr <0, -1, y_hat_tr)
    y_hat_tr = np.where(y_hat_tr >0, 1, y_hat_tr)

    err_tr = np.count_nonzero(y_tr - y_hat_tr) / len(y_tr)
    
    # Compute estimate and classification error for training set
    y_hat_te = X_te.dot(z_hat)
    # convert y_hat entries to -1 and 1
    y_hat_te = np.where(y_hat_te <0, -1, y_hat_te)
    y_hat_te = np.where(y_hat_te >0, 1, y_hat_te)
    err_te = np.count_nonzero(y_te - y_hat_te) / len(y_te)
    
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
    return res


''' exercise 2
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([2, 3, 5, 7, 11, 13])
m = 2

# Create Vandermonde matrix A
A = create_vandermonde(x,m)

# Use linear algebra to solve least-squares problem and minimize || y - A z ||^2
z_hat = solve_linear_LS(A,y)

# Compute the mean squared error
f = A.dot(z_hat)
error = y - f
mse = 1/len(y) * np.inner(error,error)

# Generate x/y plot points for the fitted polynomial
xx = np.linspace(0, 7)
Vxx = create_vandermonde(xx,m) # Vandermonde
yy = Vxx.dot(z_hat)

plt.figure(figsize=(8, 4))
plt.scatter(x, y, color='red', label='data points')
plt.plot(xx, yy, linestyle='dotted',label='normal equation poly fit')
plt.legend()

poly1_expr = ' + '.join(['{0:.4f} x^{1}'.format(v, i) for i, v in enumerate(z_hat)][::-1])[:-4]
print('normal equation polynomial fit is {0}'.format(poly1_expr))
print('normal equation MSE is {0:.4f}'.format(mse))
'''

'''exercse 3'''
'''
step = 0.0002
z2_hat, mse2, T = solve_linear_LS_gd(A, y, step, 10000)
'''

# Read MNIST csv file into dataframe
df = pd.read_csv('mnist_train.csv')
# append feature column by merging all pixel columns
df['feature'] = df.apply(lambda row: row.values[1:], axis=1)
# only keep feature and label column
df = df[['feature', 'label']]

'''exercise 4'''
# X_tr, X_te, y_tr, y_te = extract_and_split(df, d=0)

# mnist_pairwise_LS(df, 0 ,1, verbose=True)

# num_trial, err_matrix = 1, np.zeros((10, 10))
# for a, b in tqdm(it.combinations(range(10), 2), total=45):
#     err_tr, err_te = np.mean([mnist_pairwise_LS(df, a, b) for _ in range(num_trial)], axis=0)
#     err_matrix[a, b], err_matrix[b, a] = err_tr, err_te

# print(np.round(err_matrix*100, 2))

# Randomly split into training/testing set
test_size = 0.5
n, m = len(df), int(len(df) * test_size)
perm = np.random.permutation(n)
tr, te = df.iloc[perm[m:]], df.iloc[perm[:m]]

# Construct the training set
X_train = tr.loc[:,'feature']
X_train = X_train.apply(lambda x: np.append(x,-1))
X_tr = np.stack(X_train.values)
y_train = tr.loc[:,'label']
y_tr = np.stack(y_train.values)
  
# Construct the testing set
X_test = te.loc[:,'feature']
X_test = X_test.apply(lambda x: np.append(x,-1))
X_te = np.stack(X_test.values)
y_test = te.loc[:,'label']
y_te = np.stack(y_test.values)

# Apply one-hot encoding to training labels
Y = np.zeros((len(y_tr),10))
Y_te = np.zeros((len(y_te),10))
I = np.eye(10) # identity matrix
for j, value in enumerate(y_tr):
    Y[j] = I[value]

for j, value in enumerate(y_te):
    Y_te[j] = I[value]


# Run least-square on training set
Z = np.linalg.lstsq(X_tr,Y)[0]

# Compute estimates and errors on training set
y_hat_train = X_tr @  Z
y_hat_tr = []
for item in y_hat_train:
    index = np.where(item==np.max(item))[0][0]
    item = I[index]
    y_hat_tr.append(item)

# compute err_tr
err = Y - y_hat_tr
counter = 0
for item in err:
    if linalg.norm(item,1) > 0:
        counter += 1
err_tr = counter / len(Y)


# Compute estimates and errors on training set
y_hat_test = X_te @ Z

y_hat_te = []
for item in y_hat_test:
    index = np.where(item==np.max(item))[0][0]
    item = I[index]
    y_hat_te.append(item)

err1 = Y_te - y_hat_te
counter_te = 0
for item in err1:
    if linalg.norm(item,1) > 0:
        counter_te += 1
err_te = counter_te / len(Y_te)

# convert y_hat_te to vector
y_hat_te_vec = []
for item in y_hat_te:
    index = np.where(item != 0)[0][0]
    y_hat_te_vec.append(index)
y_hat_te_vec = np.array(y_hat_te_vec)

print('training error = {0:.2f}%, testing error = {1:.2f}%'.format(100 * err_tr, 100 * err_te))
# Compute confusion matrix
cm = np.zeros((10, 10), dtype=np.int64)
for a in range(10):
    for b in range(10):
        cm[a, b] = ((y_te == a) & (y_hat_te_vec == b)).sum()
print('Confusion matrix:\n {0}'.format(cm))

## new line
zzz = 1

