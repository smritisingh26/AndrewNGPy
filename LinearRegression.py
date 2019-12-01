import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
data = pd.read_csv('ex1data1.txt', header = None)
X = data.iloc[:,0]
y = data.iloc[:,1]
m = len(y)
print(data.head())
plt.scatter(X, y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
#plt.show()
X = X[:,np.newaxis]
y = y[:,np.newaxis]
theta = np.zeros([2,1])
iterations = 1500
alpha = 0.01
ones = np.ones((m,1))
X = np.hstack((ones, X))
def computeCost(X, y, theta):
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp, 2)) / (2*m)

J = computeCost(X, y, theta)
#print(J)

def gradientDescent(X, y, theta, alpha, iterations):
    for _ in range(iterations):
        temp = np.dot(X, theta) - y
        temp = np.dot(X.T, temp)
        theta = theta - (alpha/m) * temp
    return theta

theta = gradientDescent(X, y, theta, alpha, iterations)
print(theta)

J = computeCost(X, y, theta)
print(J)

plt.scatter(X[:,1], y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1], np.dot(X, theta))
plt.show()
