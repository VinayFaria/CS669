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
#print(list_of_each_pair1) 
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
#print(list_of_each_pair2)
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
#Estimation of class
class1_classified1 = []
class1_notclassified = []
class1_classified2 = []
class2_classified2 = []
class2_notclassified = []
class2_classified1 = []
for i in testdata1:
    if i[0]>min1_clm1 and i[0]<max1_clm1 and i[1]>min1_clm2 and i[1]<max1_clm2:
        class1_classified1.append(i)
    elif i[0]>min2_clm1 and i[0]<max2_clm1 and i[1]>min2_clm2 and i[1]<max2_clm2:
        class1_classified2.append(i)
    else:
        class1_notclassified.append(i)

for i in testdata2:
    if i[0]>min1_clm1 and i[0]<max1_clm1 and i[1]>min1_clm2 and i[1]<max1_clm2:
        class2_classified1.append(i)
    elif i[0]>min2_clm1 and i[0]<max2_clm1 and i[1]>min2_clm2 and i[1]<max2_clm2:
        class2_classified2.append(i)
    else:
        class2_notclassified.append(i)

#Confusion matrix
matrix_for_number_of_elements = [[len(class1_classified1),len(class1_classified2)],[len(class2_classified1),len(class2_classified2)],[len(class1_notclassified),len(class2_notclassified)]]
accuracy = (matrix_for_number_of_elements[0][0] + matrix_for_number_of_elements[1][1])*100/(matrix_for_number_of_elements[0][0] + matrix_for_number_of_elements[0][1] + matrix_for_number_of_elements[1][0] + matrix_for_number_of_elements[1][1] + matrix_for_number_of_elements[2][0] + matrix_for_number_of_elements[2][1])
print('Accuracy is ',accuracy,'%')
confusion_dict = {'':["class1","class2"],'classified1':[len(class1_classified1),len(class2_classified1)],'classified2':[len(class1_classified2),len(class2_classified2)],'not classified1':[len(class1_notclassified),len(class2_notclassified)]}
# creating a dataframe from a dictionary
df = pd.DataFrame(confusion_dict)
print(df)

#==================================================================
#plotting histogram
#bins1 = round((max1_clm1-min1_clm1)/20)
#bins2 = round((max2_clm1-min2_clm1)/20)
f = plt.figure(1)
plt.hist(feature1_class1_arr, density=True, bins='auto', color ="cyan", alpha=0.5, label="feature1 class1")
plt.hist(feature1_class2_arr, density=True, bins='auto', color ="red", alpha=0.5, label="feature1 class2")
plt.xlim([min(min1_clm1,min2_clm1), max(max1_clm1, max2_clm1)])
plt.legend()
plt.ylabel('Probability')
plt.xlabel('feature1_numbers')
f.show()

g = plt.figure(2)
plt.hist(feature2_class1_arr, density=True, bins = 50, color ="magenta", alpha=0.5, label="feature2 class1")
plt.hist(feature2_class2_arr, density=True, bins = 50, color ="brown", alpha=0.5, label="feature2 class2")
plt.xlim([min(min1_clm2,min2_clm2), max(max1_clm2, max2_clm2)])
plt.legend()
plt.ylabel('Probability')
plt.xlabel('feature2_numbers')
g.show()