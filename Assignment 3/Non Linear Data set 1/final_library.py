import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

# Counting number of data
count = 0
with open("class2.txt") as myfile:
    for i in myfile:
        count += 1 #counting number of data
        
dimension = 2 # dimension of the class

# storing raw data in array
index = 0
X = np.zeros(shape = (int(count), int(dimension)), dtype = float)
with open("class2.txt") as myfile:
    for i in myfile:
        i = i.split(',')
        X[index][0] = float(i[0])
        X[index][1] = float(i[1])
        index += 1
"""
# 0. Create dataset
X,Y = make_blobs(cluster_std=0.5,random_state=20,n_samples=1000,centers=5)

# Stratch dataset to get ellipsoid data
X = np.dot(X,np.random.RandomState(0).randn(2,2))
"""
x,y = np.meshgrid(np.sort(X[:,0]),np.sort(X[:,1]))
XY = np.array([x.flatten(),y.flatten()]).T

GMM = GaussianMixture(n_components=2).fit(X) # Instantiate and fit the model
print('Converged:',GMM.converged_) # Check if the model has converged
means = GMM.means_ 
covariances = GMM.covariances_


# Predict
Y = np.array([[0.5],[0.5]])
prediction = GMM.predict_proba(Y.T)
print(prediction)

# Plot   
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
ax0.scatter(X[:,0],X[:,1])
ax0.scatter(Y[0,:],Y[1,:],c='orange',zorder=10,s=100)
for m,c in zip(means,covariances):
    multi_normal = multivariate_normal(mean=m,cov=c)
    ax0.contour(np.sort(X[:,0]),np.sort(X[:,1]),multi_normal.pdf(XY).reshape(len(X),len(X)),colors='black',alpha=0.3)
    ax0.scatter(m[0],m[1],c='grey',zorder=10,s=100)
    
plt.show()