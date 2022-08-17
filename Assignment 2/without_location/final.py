#=============================================================================
# Loading library
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

#=============================================================================
# Starting timer
tic = time.perf_counter()

#=============================================================================
# Defining dimenesion and number of cluster
dimension = 3  # 3 for only pixel colour values as features i.e. RGB
K = 10        # take general value (number of different clusters)

#=============================================================================
# Converting image pixel RGB to array
# The return value, img1, is a PIL image object.
img = Image.open('Image.jpg')

# When using Image.convert('RGB') it just converts each pixel to the triple 8-bit value.
img = img.convert("RGB")
image_pixel_array = np.array(img)

# record the original shape
shape = image_pixel_array.shape
print("original shape of Image : ", image_pixel_array.shape)

# make a 1-dimensional view of arr
horizontal_array = image_pixel_array.ravel()
# asarray changes input array while array makes copy
vertical_arr = np.array(horizontal_array).reshape(image_pixel_array.shape[0]*image_pixel_array.shape[1] ,3)
number_of_pixel = vertical_arr.shape[0]
print(vertical_arr.shape)
img.close()

"""
#=============================================================================
# Converting vertical array to original shape for checking the data which will
# we operate is of correct form
# reforming a numpy array of the original shape
arr2 = np.asarray(vertical_arr).reshape(shape)
#print(arr2)

# make a PIL image
img2 = Image.fromarray(arr2, 'RGB')
img2.show() 
"""
#=============================================================================
# Finding a particular pixel to which cluster it belong to
def cluster_number(Xn, means):
    Cluster_no = 0
    cluster_assigned = 0
    Minimum_distance = 10**8
    for mean in means:
        distance =0
          
        distance  = pow(np.linalg.norm(Xn-mean),2) # square of norm
        
        if distance < Minimum_distance:
            Minimum_distance =  distance
            cluster_assigned = Cluster_no
        Cluster_no += 1
    return cluster_assigned

#=============================================================================
# Finding mean of each cluster
def new_mean(r_nk, Data, Cluster_no):
    mean = np.zeros(shape=(int(dimension)),dtype =float)
    sum = np.zeros(shape=(int(dimension)),dtype =float)
    j =  0
    samples_in_cluster = 0
    for i in Data:
        if r_nk[j][Cluster_no] == 1:
            sum = sum + i
            samples_in_cluster += 1
                
        j += 1
    
    if samples_in_cluster != 0:
        mean = np.divide(sum,samples_in_cluster)
    return mean

#=============================================================================
# Finding distortion measure
def distortion_measure(r_nk, Data):
    loss = 0
    j =0
    for i in Data:
        for dummy in range(K):
            if r_nk[j][dummy]==1:
                loss += pow(np.linalg.norm(i- means[dummy]),2)
    
        j += 1
    return loss

#=============================================================================
# optimising each pixel according to cluster
means = np.random.uniform(low=0 , high =255 , size=(int(K), int(dimension)))
iter_num = 0  # Iteration Number
error = 0 # error/cost/loss
preerror = 0 # error of previous Iteration.

while True:
    j = 0
    preerror = error
    print("\nIteration number: ",iter_num)
    r_nk = np.zeros(shape=(int(number_of_pixel), int(K)), dtype=float)
    new_vertical_arr = np.zeros(shape=(int(number_of_pixel), int(dimension)), dtype=float)

    for i in vertical_arr:
        PredictCluster = cluster_number(i, means)
        
        # choosing rnk to be 1 among K clusters
        for dummy in range(K):
            if dummy == PredictCluster:
                r_nk[j][dummy] = 1
            else:
                r_nk[j][dummy] = 0
        
        # Divide by 255 to get range within [0, 1]. This is required Only for 
        # plotting an RGB image.
        # The reason is that if the color intensity is a float, then matplotlib
        # expects it to range from 0 to 1. If an int, then it expects 0 to 255.
        new_vertical_arr[j] = means[PredictCluster]/255
        
        j += 1

    new_image_pixel_array = new_vertical_arr.reshape(image_pixel_array.shape[0], image_pixel_array.shape[1], image_pixel_array.shape[2])

    # Update the mean value for each cluster
    for i in range(0,K):
        means[i] = new_mean(r_nk, vertical_arr, i)

    name = "plot/K10/Iteration" + str(iter_num)
    plt.imsave(name + '.png', new_image_pixel_array)
    error = distortion_measure(r_nk, vertical_arr)
    print("preerror: ",preerror)
    print("error: ", error)

    # If further there is no change error or threshold iteration is reached
    if iter_num ==  40 or error == preerror:
        toc = time.perf_counter()   # stopping timer
        print(f'Time taken: {toc - tic:0.4f} seconds')
        break
    iter_num += 1