import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#==================================================================
#reading class1 file
class1=open('class1.txt','r')
details1=class1.readlines()
#print(details1)
list_of_each_pair1=[]
for i in details1:
    #print(i)
    data=i.strip()
    x1,y1=data.split(',')
    list_of_each_pair1.append((float(x1),float(y1)))
#==================================================================
#reading class2 file
class2=open('class2.txt','r')
details2=class2.readlines()
#print(details2)
list_of_each_pair2=[]
for j in details2:  
    #print(i)    
    data=j.strip()
    x2,y2=data.split(',')
    list_of_each_pair2.append((float(x2),float(y2)))
#==================================================================
#splitting the data in 80:20
testdata1 = []
testdata2 = []
trainingdata1 = []
trainingdata2 = []
for i in range(int(len(list_of_each_pair1))):
    if i%5 == 0:
        testdata1.append(list_of_each_pair1[i])
    else:
        trainingdata1.append(list_of_each_pair1[i])
        
for i in range(int(len(list_of_each_pair2))):
    if i%5 == 0:
        testdata2.append(list_of_each_pair2[i])
    else:
        trainingdata2.append(list_of_each_pair2[i])
#==================================================================
#finding minimum and maximum of column1 and column2 of class1
min1_clm1 = trainingdata1[0][0]
feature1_class1 = []
feature2_class1 = []
for i in trainingdata1:
    feature1_class1.append(i[0])
    feature2_class1.append(i[1])
    if i[0] < min1_clm1:
        min1_clm1 = i[0]
    else:
        continue
#print(min1_clm1)
feature1_class1_arr = np.asarray(feature1_class1)
feature2_class1_arr = np.asarray(feature2_class1)
max1_clm1 = trainingdata1[0][0]
for i in trainingdata1:
    if i[0] > max1_clm1:
        max1_clm1 = i[0]
    else:
        continue
#print(max1_clm1)
min1_clm2 = trainingdata1[0][1]
for i in trainingdata1:
    if i[1] < min1_clm2:
        min1_clm2 = i[1]
    else:
        continue
#print(min1_clm2)
max1_clm2 = trainingdata1[0][1]
for i in trainingdata1:
    if i[1] > max1_clm2:
        max1_clm2 = i[1]
    else:
        continue
#print(max1_clm2)

#==================================================================
#finding minimum and maximum of column1 and column2 of class2    
min2_clm1 = trainingdata2[0][0]
feature1_class2 = []
feature2_class2 = []
for i in trainingdata2:
    feature1_class2.append(i[0])
    feature2_class2.append(i[1])
    if i[0] < min2_clm1:
        min2_clm1 = i[0]
    else:
        continue
#print(min2_clm1)
feature1_class2_arr = np.asarray(feature1_class2)
feature2_class2_arr = np.asarray(feature2_class2)
max2_clm1 = trainingdata2[0][0]
for i in trainingdata2:
    if i[0] > max2_clm1:
        max2_clm1 = i[0]
    else:
        continue
#print(max2_clm1)    
min2_clm2 = trainingdata2[0][1]
for i in trainingdata2:
    if i[1] < min2_clm2:
        min2_clm2 = i[1]
    else:
        continue
#print(min2_clm2)    
max2_clm2 = trainingdata2[0][1]
for i in trainingdata2:
    if i[1] > max2_clm2:
        max2_clm2 = i[1]
    else:
        continue
#print(max2_clm2)

#==================================================================
#Bins for feature of classes
def bin_array(min_data,max_data,number_of_datapoints):
    number_of_bins = pow(number_of_datapoints , 0.5)
    binwidth = float((max_data-min_data)/number_of_bins)
    bins = np.arange(min_data , max_data + binwidth , binwidth)
    return bins

bins_class1_feature1 = bin_array(min1_clm1, max2_clm1, len(trainingdata1))
#bins_class1_feature1 = bins_class1_feature1.tolist()
#print(bins_class1_feature1)
bins_class1_feature2 = bin_array(min1_clm2, max1_clm2, len(trainingdata1))
#bins_class1_feature2 = bins_class1_feature2.tolist()
#print(bins_class1_feature2)
bins_class2_feature1 = bin_array(min2_clm1, max2_clm1, len(trainingdata2))
#bins_class2_feature1 = bins_class2_feature1.tolist()
#print(bins_class2_feature1)
bins_class2_feature2 = bin_array(min2_clm2, max2_clm2, len(trainingdata2))
#bins_class2_feature2 = bins_class2_feature2.tolist()
#print(bins_class2_feature2)

#==================================================================
#Calculating probability
def hist_feature1(bins_class_feature, trainingdata):
    hist_dict = {}
    for i in range(len(bins_class_feature)-1):
        hist_dict[(bins_class_feature[i],bins_class_feature[i+1])] = 0
        for j in trainingdata:
            if j[0] >= bins_class_feature[i] and j[0] < bins_class_feature[i+1]:
                hist_dict[(bins_class_feature[i],bins_class_feature[i+1])] += 1
            else:
                continue
    return hist_dict

def hist_feature2(bins_class_feature, trainingdata):
    hist_dict = {}
    for i in range(len(bins_class_feature)-1):
        hist_dict[(bins_class_feature[i],bins_class_feature[i+1])] = 0
        for j in trainingdata:
            if j[1] >= bins_class_feature[i] and j[1] < bins_class_feature[i+1]:
                hist_dict[(bins_class_feature[i],bins_class_feature[i+1])] += 1
            else:
                continue
    return hist_dict
def normalization(hist_dict_feature):
    factor=1.0/sum(hist_dict_feature.values())
    normalised_dict_feature = {k: v*factor for k, v in hist_dict_feature.items()}
    return normalised_dict_feature

# dictionary for number of times occurance of element in class feature
hist_dict1_feature1 = hist_feature1(bins_class1_feature1, trainingdata1)
#print(hist_dict1_feature1)
hist_dict1_feature2 = hist_feature2(bins_class1_feature2, trainingdata1)
#print(hist_dict1_feature2)
hist_dict2_feature1 = hist_feature1(bins_class2_feature1, trainingdata2)
#print(hist_dict2_feature1)
hist_dict2_feature2 = hist_feature2(bins_class2_feature2, trainingdata2)
#print(hist_dict2_feature2)

# normalised dictionary
normalised_dict1_feature1 = normalization(hist_dict1_feature1)
normalised_dict1_feature2 = normalization(hist_dict1_feature2)
normalised_dict2_feature1 = normalization(hist_dict2_feature1)
normalised_dict2_feature2 = normalization(hist_dict2_feature2)
print(normalised_dict2_feature2)

#==================================================================
#Test data classification
def classify_data(data, normalised_dict1_feature1, normalised_dict1_feature2, normalised_dict2_feature1, normalised_dict2_feature2):
    p1_f1 = 0
    p1_f2 = 0
    for key,value in normalised_dict1_feature1.items():
        if (data[0] >= key[0]) and (data[0] < key[1]):
            #assigning probability of class1_feature1 probability
            p1_f1 = value
        else:
            continue
    for key,value in normalised_dict1_feature2.items():
        if (data[1] >= key[0]) and (data[1] < key[1]):
            #assigning probability of class1_feature2 probability
            p1_f2 = value
        else:
            continue
    probability_tobe_class1 = p1_f1*p1_f2
    
    p2_f1 = 0
    p2_f2 = 0
    for key,value in normalised_dict2_feature1.items():
        if (data[0] >= key[0]) and (data[0] < key[1]):
            #assigning probability of class1_feature1 probability
            p2_f1 = value
        else:
            continue
    for key,value in normalised_dict2_feature2.items():
        if (data[1] >= key[0]) and (data[1] < key[1]):
            #assigning probability of class1_feature2 probability
            p2_f2 = value
        else:
            continue
    probability_tobe_class2 = p2_f1*p2_f2

    if probability_tobe_class1 == 0 and probability_tobe_class2 == 0:
        return 'notclassified'
    elif probability_tobe_class1 > probability_tobe_class2:
        return 'class1'
    elif probability_tobe_class1 < probability_tobe_class2:
        return 'class2'
    
#==================================================================
#Estimation of class
testdata_class1=[]
class1_classified1 = []
class1_classified2 = []
class1_notclassified = []
testdata_class2=[]
class2_classified2 = []
class2_classified1 = []
class2_notclassified = []

for i in testdata1:
    output = classify_data(i, normalised_dict1_feature1, normalised_dict1_feature2, normalised_dict2_feature1, normalised_dict2_feature2)
    if output == 'class1':
        testdata_class1.append(i)
        class1_classified1.append(i)
    elif output == 'class2':
        testdata_class2.append(i)
        class1_classified2.append(i)
    elif output == 'notclassified':
        class1_notclassified.append(i)

for i in testdata2:
    output = classify_data(i, normalised_dict1_feature1, normalised_dict1_feature2, normalised_dict2_feature1, normalised_dict2_feature2)
    if output == 'class1':
        testdata_class1.append(i)
        class2_classified1.append(i)
    elif output == 'class2':
        testdata_class2.append(i)
        class2_classified2.append(i)
    elif output == 'notclassified':
        class2_notclassified.append(i)

#==================================================================
#Confusion matrix
matrix_for_number_of_elements = [[len(class1_classified1),len(class1_classified2)],[len(class2_classified1),len(class2_classified2)],[len(class1_notclassified),len(class2_notclassified)]]
accuracy = (matrix_for_number_of_elements[0][0] + matrix_for_number_of_elements[1][1])*100/(matrix_for_number_of_elements[0][0] + matrix_for_number_of_elements[0][1] + matrix_for_number_of_elements[1][0] + matrix_for_number_of_elements[1][1] + matrix_for_number_of_elements[2][0] + matrix_for_number_of_elements[2][1])
print('For non-linear separable data Accuracy is ',accuracy,'%')
confusion_dict = {'':["class1","class2"],'classified1':[len(class1_classified1),len(class2_classified1)],'classified2':[len(class1_classified2),len(class2_classified2)],'not classified':[len(class1_notclassified),len(class2_notclassified)]}
# creating a dataframe from a dictionary
df = pd.DataFrame(confusion_dict)
print(df)

"""
#==================================================================
#plotting histogram
testdata_feature1 = []
testdata_feature2 = []
for i in testdata_class1:
    testdata_feature1.append(i[0])
    testdata_feature2.append(i[1])

f = plt.figure(1)
plt.hist(feature1_class1_arr, bins=bins_class1_feature1 , color ="cyan", alpha=0.5, label="feature1 class1")
plt.hist(feature1_class2_arr, bins=bins_class2_feature1 , color ="red", alpha=0.5, label="feature1 class2")
#plt.hist(testdata_feature1, bins= 'auto', color ="green", alpha=0.5, label="testdata_feature1")
plt.xlim([min(min1_clm1,min2_clm1), max(max1_clm1, max2_clm1)])
plt.legend()
plt.ylabel('Occurrence')
plt.xlabel('feature1_numbers')
f.show()

g = plt.figure(2)
plt.hist(feature2_class1_arr, bins = bins_class1_feature2, color ="magenta", alpha=0.5, label="feature2 class1")
plt.hist(feature2_class2_arr, bins = bins_class2_feature2, color ="brown", alpha=0.5, label="feature2 class2")
#plt.hist(testdata_feature2, bins= 'auto', color ="green", alpha=0.5, label="testdata_feature2")
plt.xlim([min(min1_clm2,min2_clm2), max(max1_clm2, max2_clm2)])
plt.legend()
plt.ylabel('Occurrence')
plt.xlabel('feature2_numbers')
g.show()
"""