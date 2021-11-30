#=============================================================================
# Loading library
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import time

#=============================================================================
# Starting timer
tic=time.perf_counter()

#=============================================================================
# Defining dimenesion and number of cluster
print("GMM")
K = 2 # choosing the K value where K denotes the number of clusters/gaussians
print('For K = ',K)
#K=int(input("Enter The Value of K:"))
dimension = 2 # dimension of the class

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
#class1
# Counting number of data
count = 0
with open("Class1.txt") as myfile:
    for i in myfile:
        count += 1 #counting number of data

#=============================================================================
# storing raw data in array
index = 0
X = np.zeros(shape = (int(count), int(dimension)), dtype = float)
with open("Class1.txt") as myfile:
    for i in myfile:
        i = i.split(',')
        X[index][0] = float(i[0])
        X[index][1] = float(i[1])
        index += 1
        
#=============================================================================
#class2
# Counting number of data
count1 = 0
with open("Class2.txt") as myfile1:
    for i in myfile1:
        count1 += 1 #counting number of data

#=============================================================================
# storing raw data in array
index1 = 0
X1 = np.zeros(shape = (int(count1), int(dimension)), dtype = float)
with open("Class2.txt") as myfile1:
    for i in myfile1:
        i = i.split(',')
        X1[index1][0] = float(i[0])
        X1[index1][1] = float(i[1])
        index1 += 1

#=============================================================================
# Set the initial mean, covariance and pi values
# Mean is a K*dimension matrix where K is number of Gaussians and is dimension
mean = np.random.randint(min(X[:,0]),max(X[:,0]),size=(K,dimension))
mean1 = np.random.randint(min(X1[:,0]),max(X1[:,0]),size=(K,dimension))

# We need a n covariance matrix of size mxm for each gaussian
covariance = np.zeros((K,dimension,dimension))
covariance1 = np.zeros((K,dimension,dimension))

for dim in range(len(covariance)):
    np.fill_diagonal(covariance[dim],1)
for dim in range(len(covariance1)):
    np.fill_diagonal(covariance1[dim],1)
    
pi = np.ones(K)/K # weight are assigned equal to every cluster
reg_cov = 1e-6*np.identity(len(X[0]))
pi1 = np.ones(K)/K # weight are assigned equal to every cluster
reg_cov1 = 1e-6*np.identity(len(X1[0]))

#=============================================================================
# Plot the initial state
x,y = np.meshgrid(np.sort(X[:,0]),np.sort(X[:,1]))
X_Y_merged = np.array([x.flatten(),y.flatten()]).T
x1,y1 = np.meshgrid(np.sort(X1[:,0]),np.sort(X1[:,1]))
X_Y_merged1 = np.array([x1.flatten(),y1.flatten()]).T
plt.figure(1)
plt.scatter(X[:,0],X[:,1])
plt.title('Initial state1')
for m,c in zip(mean,covariance):
    c += reg_cov
    multi_normal = multivariate_normal(mean=m,cov=c)
    plt.contour(np.sort(X[:,0]),np.sort(X[:,1]),multi_normal.pdf(X_Y_merged).reshape(len(X),len(X)),colors='black',alpha=0.3)
    plt.scatter(m[0],m[1],c='purple',zorder=10,s=50)
plt.scatter(X1[:,0],X1[:,1])
#plt.title('Initial state2')
for m,c in zip(mean1,covariance1):
    c += reg_cov1
    multi_normal1 = multivariate_normal(mean=m,cov=c)
    plt.contour(np.sort(X1[:,0]),np.sort(X1[:,1]),multi_normal.pdf(X_Y_merged1).reshape(len(X1),len(X1)),colors='black',alpha=0.3)
    plt.scatter(m[0],m[1],c='purple',zorder=10,s=50)
    
#=============================================================================
iter_num = 0
responsibility = np.zeros((len(X),K))
responsibility1 = np.zeros((len(X1),K))
log_likelihood = []
log_likelihood1 = []
#for i in range(iterations):
while True:

    # E Step
    responsibility = E_step(mean, covariance, pi, K)
    responsibility1 = E_step(mean1, covariance1, pi1, K)

    # M Step
    mean, covariance, pi = M_step(responsibility)
    mean1, covariance1, pi1 = M_step(responsibility1)

    """Log likelihood"""
    log_likelihood.append(log_likeli(mean, covariance, pi))
    log_likelihood1.append(log_likeli(mean1, covariance1, pi1))

    if iter_num == 45:
        """
        kk=1
        for k in mean:
            plt.plot(k[0],k[1],"o",c='black')
            plt.text(k[0],k[1],'Mean {}'.format(kk))
            kk=kk+1
        kk1=1
        for k in mean1:
            plt.plot(k[0],k[1],"o",c='black')
            plt.text(k[0],k[1],'Mean {}'.format(kk1))
            kk1=kk1+1
        """
        toc = time.perf_counter()   # stopping timer
        print(f'Time taken: {toc - tic:0.4f} seconds')  # printing the execution time
        break
    iter_num +=1

mean_cluster1 = mean
covariance_cluster1 = covariance
mean_cluster2 = mean1
covariance_cluster2 = covariance1

#=============================================================================
# Counting number of data
count = 0
with open("class1_class2.txt") as myfile:
    for i in myfile:
        count += 1 #counting number of data
    
#=============================================================================
# storing test data in list
index = 0
test = np.zeros(shape = (int(count), int(dimension)), dtype = float)
with open("class1_class2.txt") as myfile:
    for i in myfile:
        i = i.split(',')
        test[index][0] = float(i[0])
        test[index][1] = float(i[1])
        index += 1

testdata_class1_cluster1=[]
testdata_class1_cluster2=[]
testdata_class2_cluster1=[]
testdata_class2_cluster2=[]
for i in test:
    t1 = multivariate_normal.pdf(i, mean=mean_cluster1[0], cov=covariance_cluster1[0])
    t2 = multivariate_normal.pdf(i, mean=mean_cluster1[1], cov=covariance_cluster1[1])
    t3 = multivariate_normal.pdf(i, mean=mean_cluster2[0], cov=covariance_cluster2[0])
    t4 = multivariate_normal.pdf(i, mean=mean_cluster2[1], cov=covariance_cluster2[1])
    if t1>t2 and t1>t3 and t1>t4:
        testdata_class1_cluster1.append(i)
    elif t2>t1 and t2>t3 and t2>t4:
        testdata_class1_cluster2.append(i)
    elif t3>t1 and t3>t2 and t3>t4:
        testdata_class2_cluster1.append(i)
    elif t4>t1 and t4>t2 and t4>t3:
        testdata_class2_cluster2.append(i)

plt.figure(2)
plt.scatter(*zip(*testdata_class1_cluster1),c="purple")
plt.scatter(*zip(*testdata_class1_cluster2),c="cyan")
plt.scatter(*zip(*testdata_class2_cluster1),c="orange")
plt.scatter(*zip(*testdata_class2_cluster2),c="pink")
plt.legend(['testdata_class1_cluster1','testdata_class1_cluster2','testdata_class2_cluster1','testdata_class2_cluster2'])
plt.title('Non-linearly seperable')
"""
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

#=============================================================================
# r is a matrix which contain probability of every datapoint wrt each cluster
# number of row = number of datapoint, number of column = number of clusters/gaussians
r = np.zeros(shape = (int(count), int(K)), dtype = float)


#Probability for each datapoint x_i to belong to gaussian g

for c,g in zip(range(3),[gauss_1,gauss_2,gauss_3]):
    r[:,c] = g.pdf(X_tot) # Write the probability that x belongs to gaussian c in column c.

"""
