#IMPORTING LIIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.linalg import svd

#LOADING DATA
mat = loadmat("ex7data1.mat")
X = mat["X"]

#PLOTTING DATA
plt.scatter(X[:,0],X[:,1],marker="o",facecolors="none",edgecolors="b")
plt.show()

#FEATURE NORMALIZATION
def FeatureNormalize(X):
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    X_norm = (X - mu)/sigma
    return X_norm, mu , sigma

#RUNNING PCA
def PCA(X):
    m,n = X.shape[0],X.shape[1]
    sigma = 1/m * X.T @ X
    U,S,V = svd(sigma)
    return U,S,V

X_norm,mu,std = FeatureNormalize(X)
U,S = PCA(X_norm)[:2]
plt.scatter(X[:,0],X[:,1],marker="o",facecolors="none",edgecolors="b")
plt.plot([mu[0],(mu+1.5*S[0]*U[:,0].T)[0]],[mu[1],(mu+1.5*S[0]*U[:,0].T)[1]],color="black",linewidth=3)
plt.plot([mu[0],(mu+1.5*S[1]*U[:,1].T)[0]],[mu[1],(mu+1.5*S[1]*U[:,1].T)[1]],color="black",linewidth=3)
plt.xlim(-1,7)
plt.ylim(2,8)
plt.show()

print("Top eigenvector U(:,1) =:",U[:,0])

#REDUCING DIMENSIONALITY
def ProjectData(X,U,K):
    m = X.shape[0]
    U_reduced = U[:,:K]
    Z = np.zeros((m,K))
    for i in range(m):
        for j in range(K):
            Z[i,j] = X[i,:] @ U_reduced[:,j]
    return Z

#PROJECTING DATASET INTO ONE DIMENSION
K=1
Z = ProjectData(X_norm, U, K)
print("Projection of the first example:",Z[0][0])

#RECONSTRUCTING DATA
def RecoverData(Z,U,K):
    m,n = Z.shape[0],U.shape[0]
    X_rec = np.zeros((m,n))
    U_reduced = U[:,:K]
    for i in range(m):
        X_rec[i,:] = Z[i,:] @ U_reduced.T
    return X_rec

X_rec  = RecoverData(Z, U, K)
print("Approximation of the first example:",X_rec[0,:])

#PLOTTING NEW DATA
plt.scatter(X_norm[:,0],X_norm[:,1],marker="o",label="Original",facecolors="none",edgecolors="b",s=15)
plt.scatter(X_rec[:,0],X_rec[:,1],marker="o",label="Approximation",facecolors="none",edgecolors="r",s=15)
plt.title("The Normalized and Projected Data after PCA")
plt.legend()
plt.show()

#RUNNING PCA ON FACIAL IMAGES
mat2 = loadmat("ex7faces.mat")
X2 = mat2["X"]

fig, ax = plt.subplots(nrows=10,ncols=10,figsize=(8,8))
for i in range(0,100,10):
    for j in range(10):
        ax[int(i/10),j].imshow(X2[i+j,:].reshape(32,32,order="F"),cmap="gray")
        ax[int(i/10),j].axis("off")
plt.show()

X_norm2 = FeatureNormalize(X2)[0]
U2 =PCA(X_norm2)[0]
U_reduced = U2[:,:36].T
fig2, ax2 = plt.subplots(6,6,figsize=(8,8))
for i in range(0,36,6):
    for j in range(6):
        ax2[int(i/6),j].imshow(U_reduced[i+j,:].reshape(32,32,order="F"),cmap="gray")
        ax2[int(i/6),j].axis("off")
plt.show()

#RECOVERING Data
K2 = 100
Z2 = ProjectData(X_norm2, U2, K2)
print("The projected data Z has a size of:",Z2.shape)

# Data reconstruction
X_rec2  = RecoverData(Z2, U2, K2)

# Visualize the reconstructed data
fig3, ax3 = plt.subplots(10,10,figsize=(8,8))
for i in range(0,100,10):
    for j in range(10):
        ax3[int(i/10),j].imshow(X_rec2[i+j,:].reshape(32,32,order="F"),cmap="gray")
        ax3[int(i/10),j].axis("off")
plt.show()
