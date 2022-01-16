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
#grouping each raw of data of class1&class2
def matrix_form(p,q):
    mat=[]
    for i in range(len(p)):
        mat.append(p[i][q])
    return mat
class1_raw1=matrix_form(list_of_each_pair1,0)
class1_raw2=matrix_form(list_of_each_pair1,1)
class2_raw1=matrix_form(list_of_each_pair2,0)
class2_raw2=matrix_form(list_of_each_pair2,1)
#==================================================================
# finding mean of each raw
mean_of_class1raw1=np.mean(class1_raw1)
mean_of_class1raw2=np.mean(class1_raw2)
mean_of_class2raw1=np.mean(class2_raw1)
mean_of_class2raw2=np.mean(class2_raw2)
#==================================================================
#find covariance
cov_of_class1raw1=np.cov(class1_raw1)
cov_of_class1raw2=np.cov(class1_raw2)
cov_of_class2raw1=np.cov(class2_raw1)
cov_of_class2raw2=np.cov(class2_raw2)
#==================================================================
#Plotting the given data class1&class2
plt.scatter(class1_raw1, class1_raw2,c='black')
plt.scatter(class2_raw1, class2_raw2,c='red')
#==================================================================
#mean in matrix form
mean_class1=np.array([mean_of_class1raw1,mean_of_class1raw2]).reshape(2,1)
mean_class2=np.array([mean_of_class2raw1,mean_of_class2raw2]).reshape(2,1)
#==================================================================
#covarience in matrix form
covmat1=np.array([cov_of_class1raw1,0,0,cov_of_class1raw2]).reshape(2,2)
covmat2=np.array([cov_of_class2raw1,0,0,cov_of_class2raw2]).reshape(2,2)
#==================================================================
#find the determinant of the covariance matrix
det_covmat1=np.linalg.det(covmat1)
det_covmat2=np.linalg.det(covmat2)
#==================================================================
#finding inverse of covariance matrix
inv_covmat1=np.linalg.inv(covmat1)
inv_covmat2=np.linalg.inv(covmat2)
#==================================================================
#calculation of discriminent function
def discriminant_function(x,u,inv_covmat,det_covmat):
    m=np.dot(inv_covmat,np.subtract(x,u))
    p=np.log(det_covmat**(-0.5))-((0.5)*np.dot((x-u).T,m))
    return p
testdata_class1=[]
class1_classified1 = []
class1_classified2 = []
testdata_class2=[]
class2_classified2 = []
class2_classified1 = []
for i in testdata1:
    x=np.array(i).reshape(2,1)
    p1=discriminant_function(x,mean_class1,inv_covmat1,det_covmat1)
    p2=discriminant_function(x,mean_class2,inv_covmat2,det_covmat2)
    if (p1-p2)>0:
        class1_classified1.append(i)
        testdata_class1.append(i)
    elif (p1-p2)<0:
        class1_classified2.append(i)
        testdata_class2.append(i)

for i in testdata2:
    x=np.array(i).reshape(2,1)
    p1=discriminant_function(x,mean_class1,inv_covmat1,det_covmat1)
    p2=discriminant_function(x,mean_class2,inv_covmat2,det_covmat2)
    if (p1-p2)>0:
        class2_classified1.append(i)
        testdata_class1.append(i)
    elif (p1-p2)<0:
        class2_classified2.append(i)
        testdata_class2.append(i)

plt.scatter(*zip(*testdata_class1),c="purple")
plt.scatter(*zip(*testdata_class2),c="cyan")
plt.legend(['class1_train','class2_train','class1_test','class2_test'])
plt.title('Lineraly separable dataset')

#Confusion matrix
matrix_for_number_of_elements = [[len(class1_classified1),len(class1_classified2)],[len(class2_classified1),len(class2_classified2)]]
accuracy = (matrix_for_number_of_elements[0][0] + matrix_for_number_of_elements[1][1])*100/(matrix_for_number_of_elements[0][0] + matrix_for_number_of_elements[0][1] + matrix_for_number_of_elements[1][0] + matrix_for_number_of_elements[1][1])
print('Accuracy is ',accuracy,'%')
confusion_dict = {'':["class1","class2"],'classified1':[len(class1_classified1),len(class2_classified1)],'classified2':[len(class1_classified2),len(class2_classified2)]}
# creating a dataframe from a dictionary
df = pd.DataFrame(confusion_dict)
print(df)