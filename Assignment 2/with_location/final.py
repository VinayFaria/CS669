#=============================================================================
# Loading library
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from PIL import Image

#=============================================================================
# Starting timer
tic = time.perf_counter()

#=============================================================================
# Defining dimenesion and number of cluster
dimension = 5  # 3 for pixel colour values as features i.e. RGB and position of pixel
K = 50        # take general value (number of different clusters)

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
rows = shape[0]
columns = shape[1]

# make a 1-dimensional view of arr
horizontal_array = image_pixel_array.ravel()
# asarray changes input array while array makes copy
vertical_arr = np.array(horizontal_array).reshape(image_pixel_array.shape[0]*image_pixel_array.shape[1] ,3)
number_of_pixel = vertical_arr.shape[0]
print(vertical_arr.shape)

j=0
feature_matrix = np.zeros(shape=(int(vertical_arr.shape[0]),int(dimension)),dtype = int )
for row_number in range(rows):
    for column_number in range(columns):
        feature_value = np.zeros(shape = (5),dtype = int)
        feature_value[0] = image_pixel_array[row_number][column_number][0]
        feature_value[1] = image_pixel_array[row_number][column_number][1]
        feature_value[2] = image_pixel_array[row_number][column_number][2]
        feature_value[3] = row_number#column_number
        feature_value[4] = column_number#row_number
        feature_matrix[j] = feature_value
        j += 1

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
# Intial random mean
means = np.random.uniform(low=0 , high =255 , size=(int(K), int(dimension)))
for a in range(K):
    for b in range(dimension):
        if b == 0 or b == 1 or b == 2:   # RGB limit
            means[a][b] = random.uniform(0, 255)
        elif b == 3:
            means[a][b] = random.uniform(0, rows) ###
        elif b == 4:
            means[a][b] = random.uniform(0, columns)###

#=============================================================================
# optimising each pixel according to cluster
iter_num = 0  # Iteration Number
error = 0 # error/cost/loss
preerror = 0 # error of previous Iteration.

while True:
    j = 0
    preerror = error
    print("\nIteration number: ",iter_num)
    r_nk = np.zeros(shape=(int(number_of_pixel), int(K)), dtype=float)
    new_vertical_arr = np.zeros(shape=(int(rows*columns),int(dimension)-2), dtype=float)
    new_vector = np.zeros(shape=(int(number_of_pixel), int(dimension)), dtype=float)

    for i in feature_matrix:
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
        for b in range(dimension):
            if b == 0 or b == 1 or b == 2:   # RGB limit
                new_vector[j][b] = means[PredictCluster][b]/255
            elif b == 3:    # X limit
                new_vector[j][b] = means[PredictCluster][b]/rows#columns
            elif b == 4:    # Y limit
                new_vector[j][b] = means[PredictCluster][b]/columns#rows

        j += 1

    j = 0
    for dummy in new_vector:
        new_vertical_arr[j][0] = dummy[0]
        new_vertical_arr[j][1] = dummy[1]
        new_vertical_arr[j][2] = dummy[2]
        j +=1

    new_image_pixel_array = new_vertical_arr.reshape(rows, columns, shape[2])

    # Update the mean value for each cluster
    for i in range(K):
        means[i] = new_mean(r_nk, feature_matrix, i)

    name = "plot/K50/Iteration" + str(iter_num)
    plt.imsave(name + '.png', new_image_pixel_array)
    error = distortion_measure(r_nk, feature_matrix)
    print("preerror: ",preerror)
    print("error: ", error)

    # If further there is no change error or threshold iteration is reached
    if iter_num ==  40 or error == preerror:
        toc = time.perf_counter()   # stopping timer
        print(f'Time taken: {toc - tic:0.4f} seconds')
        break
    iter_num += 1
