#=============================================================================
# Loading library
import numpy as np
import matplotlib.pyplot as plt
import time
import colorsys

#=============================================================================
# Starting timer
tic=time.perf_counter()

#=============================================================================
# Defining dimenesion and number of cluster
print("K MEANS CLUSTERING")
K = 6 #choosing the K value
print('For K = ',K)
#K=int(input("Enter The Value of K:"))
dimension = 2 # dimension of the class

#=============================================================================
# Finding a particular data to which cluster it belong to
def cluster_number(Xn, means):
    Cluster_no = 0
    final_cluster = 0
    min_dist = 10**8
    for i in means:
        norm=pow(np.linalg.norm(Xn-i),2)
        if norm < min_dist:
            min_dist =  norm
            final_cluster = Cluster_no
        Cluster_no += 1
    return final_cluster

#=============================================================================
# Finding new mean of each cluster
def new_mean(r_nk, Data, Cluster_no):
    mean = np.zeros(shape=(int(dimension)),dtype =float)
    sum = np.zeros(shape=(int(dimension) ),dtype =float)
    j =  0
    cluster_count = 0
    for i in Data:
        if r_nk[j][Cluster_no] == 1:
            sum += i
            cluster_count += 1
        j += 1

    if cluster_count != 0:
        mean = np.divide(sum,cluster_count)
    return mean

#=============================================================================
# Finding distortion measure
def distortion_measure(r_nk, Data):
    loss = 0
    for i in range(count):
        for j in range(K):
            if r_nk[i][j]==1:
                loss+=pow(Data[i][0]-means[j][0],2)+pow(Data[i][1]-means[j][1],2)
                #cost+=pow(np.linalg.norm(Data[i]-means[j]),2)
    return loss

#=============================================================================
# K unique colors
def _get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

#=============================================================================
# Counting number of data
count = 0
with open("class2.txt") as myfile:
    for i in myfile:
        count += 1 #counting number of data

#=============================================================================
# storing raw data in array
index = 0
Data = np.zeros(shape = (int(count), int(dimension)), dtype = float)
with open("class2.txt") as myfile:
    for i in myfile:
        i = i.split(',')
        Data[index][0] = float(i[0])
        Data[index][1] = float(i[1])
        index += 1

#=============================================================================        
# Finding minimum and maximum for choosing initial mean
mean1 = 0
mean2 = 0
for i in Data:
    mean1 += i[0]
    mean2 += i[1]
mean1=mean1/len(Data)
mean2=mean2/len(Data)

minimum = min(mean1,mean2)
maximum = max(mean1,mean2)
means = np.random.uniform(low=minimum , high =maximum, size=(int(K), int(dimension)))
"""
initial_mean1 = Data[0]
initial_mean2 = Data[1]
for vector in Data:
    if np.linalg.norm(vector)<np.linalg.norm(initial_mean1):
        initial_mean1 = vector
    
    if np.linalg.norm(vector)>np.linalg.norm(initial_mean2):
        initial_mean2 = vector
print(initial_mean1)
print(initial_mean2)
mean1 = np.array([initial_mean1[0],initial_mean2[1]])
mean2 = np.array([initial_mean2[0],initial_mean1[1]])
"""
#=============================================================================
# optimising each data point according to cluster
r_nk = np.zeros(shape=(int(count), int(K)), dtype=float)
error = 0
preerror = 0
j = 0
iter_num =0

colors = _get_colors(K) # a list of K number of colours will made here

while True:
    j = 0
    preerror = error
    print("\nIteration number: ",iter_num)
    r_nk = np.zeros(shape=(int(count), int(K)), dtype=float)
    
    # classifying data in clusters based on mean
    for i in Data:
        PredictCluster = cluster_number(i, means)
        
        r_nk[j][PredictCluster] = 1
        
        j += 1
        plt.plot(float(i[0]), float(i[1]),"o",color=colors[PredictCluster])
    
    # saving plot for each iteration
    plot_name = 'Iteration' + str(iter_num) + '.png'
    plt.title("Iteration number " + str(iter_num))
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig('./plot/class2/K6/'+ plot_name)
    
    # new mean calculation
    for i in range(0,K):
        means[i] = new_mean(r_nk, Data, i)
    
    # new error calculation
    error = distortion_measure(r_nk, Data)
    print("preerror: ", preerror)
    print("error: ", error)
    
    # condition for iteration limit or no change in distortion measure
    if preerror == error or iter_num == 45:
        kk=1
        for k in means:
            plt.plot(k[0],k[1],"o",c='black')
            plt.text(k[0],k[1],'Mean {}'.format(kk))
            kk=kk+1
        toc = time.perf_counter()   # stopping timer
        print(f'Time taken: {toc - tic:0.4f} seconds')  # printing the execution time
        break
    iter_num +=1