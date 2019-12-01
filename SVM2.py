#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import SVC

#LOADING DATA
mat=loadmat("ex6data2.mat")
X=mat["X"]
y=mat["y"]

#PLOTTING DATA
m,n = X.shape[0],X.shape[1]
pos,neg = (y==1).reshape(m,1),(y==0).reshape(m,1)
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+",s=50)
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],c="b",marker="o",s=50)
plt.xlim(0,1)
plt.ylim(0.4,1)
plt.show()

#PLOTTING DECISION BOUNDARY
classifier=SVC(kernel="rbf",gamma=30)
classifier.fit(X,y.ravel())
plt.figure(figsize=(8,6))
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+")
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],c="b",marker="o")
X_1,X_2 = np.meshgrid(np.linspace(X[:,0].min(),X[:,1].max(),num=100),np.linspace(X[:,1].min(),X[:,1].max(),num=100))
plt.contour(X_1,X_2,classifier.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="g")
plt.xlim(0,1)
plt.ylim(0.4,1)
plt.show()
