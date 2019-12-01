#IMPORTING NECESSARY LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

#LOADING DATA
data = pd.read_csv('ex2data1.txt', header = None)
X = data.iloc[:,:-1]
y = data.iloc[:,2]
data.head()
#print(data.head())

#PLOTTING INITIAL DATA
mask = y == 1
adm = plt.scatter(X[mask][0].values, X[mask][1].values)
not_adm = plt.scatter(X[~mask][0].values, X[~mask][1].values,color="red")
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
plt.show()

#DEFINING SIGMOID FUNCTION
def sigmoid(x):
  return 1/(1+np.exp(-x))

#DEFINING COST FUNCTION
def costFunction(theta, X, y):
    J = (-1/m) * np.sum(np.multiply(y, np.log(sigmoid(X @ theta)))
        + np.multiply((1-y), np.log(1 - sigmoid(X @ theta))))
    return J
#DEFINING GRADIENT DESCENT
def gradient(theta, X, y):
    return ((1/m) * X.T @ (sigmoid(X @ theta) - y))

(m, n) = X.shape
X = np.hstack((np.ones((m,1)), X))
y = y[:, np.newaxis]
theta = np.zeros((n+1,1))

#CALCULATING COST FUNCTION
J = costFunction(theta, X, y)
print(J)

#OPTIMISING PARAMETERS USING fmin_tnc
temp = opt.fmin_tnc(func = costFunction,
                    x0 = theta.flatten(),fprime = gradient,
                    args = (X, y.flatten()))
print(theta_optimized)

#CALCULATING NEW COST FUNCTION
J = costFunction(theta_optimized[:,np.newaxis], X, y)
print(J)

#PLOTTING DECISION BOUNDARY
plot_x = [np.min(X[:,1]-2), np.max(X[:,2]+2)]
plot_y = -1/theta_optimized[2]*(theta_optimized[0]
          + np.dot(theta_optimized[1],plot_x))

mask = y.flatten() == 1
adm = plt.scatter(X[mask][:,1], X[mask][:,2])
not_adm = plt.scatter(X[~mask][:,1], X[~mask][:,2],color="red")
decision_boun = plt.plot(plot_x, plot_y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
plt.show()

#CONSTRUCTING FUNCTION THAT CALCULATES ACCURACY OF MODEL
def accuracy(X, y, theta, cutoff):
    pred = [sigmoid(np.dot(X, theta)) >= cutoff]
    acc = np.mean(pred == y)
    print(acc * 100)

accuracy(X, y.flatten(), theta_optimized, 0.5)
