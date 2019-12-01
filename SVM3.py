#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import SVC

#LOADING DATA
mat=loadmat("ex6data3.mat")
X=mat["X"]
y=mat["y"]
Xval = mat["Xval"]
yval = mat["yval"]

#PLOTTING DATA
m,n = X.shape[0],X.shape[1]
pos,neg = (y==1).reshape(m,1),(y==0).reshape(m,1)
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+",s=50)
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],c="b",marker="o",s=50)
plt.show()

#DEFINING HYPERPLANE
def DatasetParams(X,y,Xval,yval,vals):
    acc = 0
    best_c=0
    best_gamma=0
    for i in vals:
        C= i
        for j in vals:
            gamma = 1/j
            classifier = SVC(C=C,gamma=gamma)
            classifier.fit(X,y)
            prediction = classifier.predict(Xval)
            score = classifier.score(Xval,yval)
            if score>acc:
                acc =score
                best_c =C
                best_gamma=gamma
    return best_c, best_gamma

#PLOTTING HYPERPLANE
vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
C, gamma = DatasetParams(X, y.ravel(), Xval, yval.ravel(),vals)
classifier = SVC(C=C,gamma=gamma)
classifier.fit(X,y.ravel())
plt.figure(figsize=(8,6))
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+",s=50)
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],c="b",marker="o",s=50)
X_7,X_8 = np.meshgrid(np.linspace(X[:,0].min(),X[:,1].max(),num=100),np.linspace(X[:,1].min(),X[:,1].max(),num=100))
plt.contour(X_7,X_8,classifier.predict(np.array([X_7.ravel(),X_8.ravel()]).T).reshape(X_7.shape),1,colors="g")
plt.xlim(-0.6,0.3)
plt.ylim(-0.7,0.5)
plt.show()
