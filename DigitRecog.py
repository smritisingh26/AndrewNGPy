#IMPORTING LIBRARIES
from scipy.io import loadmat
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

#LOADING DATA
data=loadmat('ex3data1.mat')
print(data)
X=data['X']
y=data['y']

#VISUALISING DATA
_, axarr = plt.subplots(10,10,figsize=(10,10))
for i in range(10):
    for j in range(10):
       axarr[i,j].imshow(X[np.random.randint(X.shape[0])].\
reshape((20,20), order = 'F'))
       axarr[i,j].axis('off')
plt.show()

#ADDING THE INTERCEPT TERM
m = len(y)
ones = np.ones((m,1))
X = np.hstack((ones, X)) #add the intercept
(m,n) = X.shape

#DEFINING SIGMOID FUNCTION
def sigmoid(z):
    return 1/(1+np.exp(-z))

#COST FUNCTION
def costFunctionReg(theta, X, y, lmbda):
    m = len(y)
    temp1 = np.multiply(y, np.log(sigmoid(np.dot(X, theta))))
    temp2 = np.multiply(1-y, np.log(1-sigmoid(np.dot(X, theta))))
    return np.sum(temp1 + temp2) / (-m) + np.sum(theta[1:]**2) * lmbda / (2*m)

#GRADIENT DESCENT
def gradRegularization(theta, X, y, lmbda):
    m = len(y)
    temp = sigmoid(np.dot(X, theta)) - y
    temp = np.dot(temp.T, X).T / m + theta * lmbda / m
    temp[0] = temp[0] - theta[0] * lmbda / m
    return temp

#optimize
lmbda = 0.1
k = 10
theta = np.zeros((k,n))
for i in range(k):
    digit_class = i if i else 10
    theta[i] = opt.fmin_cg(f = costFunctionReg, x0 = theta[i],  fprime = gradRegularization, args = (X, (y == digit_class).flatten(), lmbda), maxiter = 50)

#PREDICT/RECOGNISE
pred = np.argmax(X @ theta.T, axis = 1)
pred = [e if e else 10 for e in pred]
print(np.mean(pred == y.flatten()) * 100)
