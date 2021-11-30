#=============================================================================
# Loading library
import matplotlib.pyplot as plt
#from sklearn.datasets import make_blobs
import numpy as np
from scipy.stats import multivariate_normal
#np.random.seed(0)
import time

#=============================================================================
# Starting timer
tic=time.perf_counter()

#=============================================================================
# Defining dimenesion and number of cluster
print("GMM")
K = 4 # choosing the K value where K denotes the number of clusters/gaussians
print('For K = ',K)
#K=int(input("Enter The Value of K:"))

#=============================================================================
# Expectation
def E_step(mu, cov, pi, K):
    gamma_z_nk = np.zeros((len(X),K))
    total = np.sum([pi_c*multivariate_normal.pdf(X, mean=mu_c, cov=cov_c) for pi_c,mu_c,cov_c in zip(pi,mu,cov)],axis=0)
    for m,c,p,r in zip(mu,cov,pi,range(K)):
        #c += reg_cov
        gamma_z_nk[:,r] = p*multivariate_normal.pdf(X, mean=m, cov=c)/total
    return gamma_z_nk

#=============================================================================
# Maximization
def M_step(gamma_z_nk):
    mean_new = []
    covariance_new = []
    pi_new = []

    for c in range(K):
        N_c = np.sum(gamma_z_nk[:,c],axis=0)
        mu_c = (1/N_c)*np.sum(X*gamma_z_nk[:,c].reshape(len(X),1),axis=0)
        mean_new.append(mu_c)

        # Calculate the covariance matrix per source based on the new mean
        covariance_new.append(((1/N_c)*np.dot((np.array(gamma_z_nk[:,c]).reshape(len(X),1)*(X-mu_c)).T,(X-mu_c)))) #+self.reg_cov)
        # Calculate pi_new which is the "fraction of points" respectively the fraction of the probability assigned to each source 
        pi_new.append(N_c/np.sum(gamma_z_nk))
    return mean_new, covariance_new, pi_new

#=============================================================================
# Log likelihood
def log_likeli(mean_new, covariance_new, pi_new):
    dummy1 = 0
    dummy2 = 0
    for i in range(len(X)):
        for j in range(K):
            dummy2 += pi_new[j]*multivariate_normal.pdf(X[i],mean_new[j], covariance_new[j])
        dummy1 += np.log(dummy2)
    return dummy1

#=============================================================================
# Counting number of data
count = 0
with open("class1_class2.txt") as myfile:
    for i in myfile:
        count += 1 #counting number of data
        
dimension = 2 # dimension of the class

#=============================================================================
# storing raw data in array
index = 0
X = np.zeros(shape = (int(count), int(dimension)), dtype = float)
with open("class1_class2.txt") as myfile:
    for i in myfile:
        i = i.split(',')
        X[index][0] = float(i[0])
        X[index][1] = float(i[1])
        index += 1

#=============================================================================
# Set the initial mu, covariance and pi values
# Mean is a nxm matrix where n is number of Gaussians and m is dimension
mean = np.random.randint(min(X[:,0]),max(X[:,0]),size=(K,dimension)) 
# We need a n covariance matrix of size mxm for each gaussian
covariance = np.zeros((K,dimension,dimension))
for i in range(K):
    np.fill_diagonal(covariance[i],5)
pi = np.ones(K)/K # weight are assigned equal to every cluster
reg_cov = 1e-6*np.identity(dimension)

#=============================================================================
# Plot the initial state
x,y = np.meshgrid(np.sort(X[:,0]),np.sort(X[:,1]))
X_Y_merged = np.array([x.flatten(),y.flatten()]).T
plt.figure(1)
plt.scatter(X[:,0],X[:,1])
plt.title('Initial state')
for m,c in zip(mean,covariance):
    c += reg_cov
    multi_normal = multivariate_normal(mean=m,cov=c)
    plt.contour(np.sort(X[:,0]),np.sort(X[:,1]),multi_normal.pdf(X_Y_merged).reshape(len(X),len(X)),colors='black',alpha=0.3)
    plt.scatter(m[0],m[1],c='purple',zorder=10,s=50)

#=============================================================================
iter_num = 0
responsibility = np.zeros((len(X),K))
log_likelihood = []
#for i in range(iterations):
while True:            

    # E Step    
    responsibility = E_step(mean, covariance, pi, K)

    # M Step
    mean, covariance, pi = M_step(responsibility)

    """Log likelihood"""
    log_likelihood.append(log_likeli(mean, covariance, pi))
    
    if iter_num == 45:
        kk=1
        for k in mean:
            plt.plot(k[0],k[1],"o",c='black')
            plt.text(k[0],k[1],'Mean {}'.format(kk))
            kk=kk+1
        
        toc = time.perf_counter()   # stopping timer
        print(f'Time taken: {toc - tic:0.4f} seconds')  # printing the execution time
        break
    iter_num +=1
    
#=============================================================================
#fig2 = plt.figure(figsize=(10,10))
#ax1 = fig2.add_subplot(111) 
plt.figure(2)
plt.title('Log-Likelihood')
plt.plot(range(0,iter_num+1,1),log_likelihood)
#plt.show()

#=============================================================================
plt.figure(3)
plt.scatter(X[:,0],X[:,1])
plt.title('Final state')
for m,c in zip(mean,covariance):
    c += reg_cov
    multi_normal = multivariate_normal(mean=m,cov=c)
    plt.contour(np.sort(X[:,0]),np.sort(X[:,1]),multi_normal.pdf(X_Y_merged).reshape(len(X),len(X)),colors='black',alpha=0.3)
    plt.scatter(m[0],m[1],c='purple',zorder=10,s=50)
"""
#=============================================================================
# r is a matrix which contain probability of every datapoint wrt each cluster
# number of row = number of datapoint, number of column = number of clusters/gaussians
r = np.zeros(shape = (int(count), int(K)), dtype = float)


#Probability for each datapoint x_i to belong to gaussian g 

for c,g in zip(range(3),[gauss_1,gauss_2,gauss_3]):
    r[:,c] = g.pdf(X_tot) # Write the probability that x belongs to gaussian c in column c. 

"""