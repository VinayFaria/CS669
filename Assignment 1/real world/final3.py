import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#============================================================
#reading class1 file
class1=open('class1.txt','r')
details1=class1.readlines()
#print(details1)
list_of_each_pair1=[]
for i in details1:  
    #print(i)    
    data=i.strip()
    x1,y1=data.split(' ') 
    list_of_each_pair1.append((float(x1),float(y1)))
#print(list_of_each_pair1) 
#===========================================================
#reading class2 file
class2=open('class2.txt','r')
details2=class2.readlines()
#print(details2)
list_of_each_pair2=[]
for j in details2:  
    #print(i)    
    data=j.strip()
    x2,y2=data.split(' ')
    list_of_each_pair2.append((float(x2),float(y2)))
#print(list_of_each_pair2)
#===========================================================
#reading class3 file
class3=open('class3.txt','r')
details3=class3.readlines()
#print(details1)
list_of_each_pair3=[]
for k in details3:  
    #print(i)    
    data=k.strip()
    x3,y3=data.split(' ')
    list_of_each_pair3.append((float(x3),float(y3)))
#print(list_of_each_pair1) 
#==========================================================
#splitting the data in 80:20
testdata1 = []
testdata2 = []
testdata3 = []
trainingdata1 = []
trainingdata2 = []
trainingdata3 = []
for i in range(int(len(list_of_each_pair1))):
    if i%5 == 0:
        testdata1.append(list_of_each_pair1[i])
    else:
        trainingdata1.append(list_of_each_pair1[i])
#print(len(testdata1))
#print(len(trainingdata1))
for i in range(int(len(list_of_each_pair2))):
    if i%5 == 0:
        testdata2.append(list_of_each_pair2[i])
    else:
        trainingdata2.append(list_of_each_pair2[i])
#print(len(testdata2))
#print(len(trainingdata2))
for i in range(int(len(list_of_each_pair3))):
    if i%5 == 0:
        testdata3.append(list_of_each_pair3[i])
    else:
        trainingdata3.append(list_of_each_pair3[i])
#print(len(testdata3))
#print(len(trainingdata3))
#===========================================================
#grouping each raw of data of class1,class2&class3
def matrix_form(p,q):
    mat=[]
    for i in range(len(p)):
        mat.append(p[i][q])
    return mat
class1_raw1=matrix_form(list_of_each_pair1,0)
class1_raw2=matrix_form(list_of_each_pair1,1)
class2_raw1=matrix_form(list_of_each_pair2,0)
class2_raw2=matrix_form(list_of_each_pair2,1)
class3_raw1=matrix_form(list_of_each_pair3,0)
class3_raw2=matrix_form(list_of_each_pair3,1)
#===========================================================
# finding mean of each raw
mean_of_class1raw1=np.mean(class1_raw1)
mean_of_class1raw2=np.mean(class1_raw2)
mean_of_class2raw1=np.mean(class2_raw1)
mean_of_class2raw2=np.mean(class2_raw2)
mean_of_class3raw1=np.mean(class3_raw1)
mean_of_class3raw2=np.mean(class3_raw2)

#===========================================================
#find covariance
cov_of_class1raw1=np.cov(class1_raw1)
cov_of_class1raw2=np.cov(class1_raw2)
cov_of_class2raw1=np.cov(class2_raw1)
cov_of_class2raw2=np.cov(class2_raw2)
cov_of_class3raw1=np.cov(class3_raw1)
cov_of_class3raw2=np.cov(class3_raw2)

#===========================================================
#Plotting the given data class1,class2&class3
plt.scatter(class1_raw1, class1_raw2,c='green')
plt.scatter(class2_raw1, class2_raw2,c='red')
plt.scatter(class3_raw1, class3_raw2,c='magenta')

#===========================================================
#mean in matrix form
mean_class1=np.array([mean_of_class1raw1,mean_of_class1raw2]).reshape(2,1)
mean_class2=np.array([mean_of_class2raw1,mean_of_class2raw2]).reshape(2,1)
mean_class3=np.array([mean_of_class3raw1,mean_of_class3raw2]).reshape(2,1)
#===========================================================
#covarience in matrix form
covmat1=np.array([cov_of_class1raw1,0,0,cov_of_class1raw2]).reshape(2,2)
covmat2=np.array([cov_of_class2raw1,0,0,cov_of_class2raw2]).reshape(2,2)
covmat3=np.array([cov_of_class3raw1,0,0,cov_of_class3raw2]).reshape(2,2)
#===========================================================
#find the determinant of the covariance matrix
det_covmat1=np.linalg.det(covmat1)
det_covmat2=np.linalg.det(covmat2)
det_covmat3=np.linalg.det(covmat3)
#===========================================================
#finding inverse of covariance matrix
inv_covmat1=np.linalg.inv(covmat1)
inv_covmat2=np.linalg.inv(covmat2)
inv_covmat3=np.linalg.inv(covmat3)
#===========================================================
#calculation of discriminent function
def discriminant_function(x,u,inv_covmat,det_covmat):
    m=np.dot(inv_covmat,np.subtract(x,u))
    p=np.log(det_covmat**(-0.5))-((0.5)*np.dot((x-u).T,m))
    return p
testdata_classified1=[]
class1_classified1 = []
class1_classified2 = []
class1_classified3 = []
testdata_classified2=[]
class2_classified2 = []
class2_classified1 = []
class2_classified3 = []
testdata_classified3=[]
class3_classified2 = []
class3_classified1 = []
class3_classified3 = []
for i in testdata1:
    x=np.array(i).reshape(2,1)
    p1=discriminant_function(x,mean_class1,inv_covmat1,det_covmat1)
    p2=discriminant_function(x,mean_class2,inv_covmat2,det_covmat2)
    p3=discriminant_function(x,mean_class3,inv_covmat3,det_covmat3)
    if (p1-p2)>0:
        if (p1-p3)>0:
            class1_classified1.append(i)
            testdata_classified1.append(i)
        else:
#            if (p3-p1)>0 or p3==p1:
            class1_classified3.append(i)
            testdata_classified3.append(i)
    elif (p2-p1)>0 or p1==p2:
        if (p2-p3)>0:
            class1_classified2.append(i)
            testdata_classified2.append(i)
        else:
#            if (p3-p2)>0 or p3==p2:
            class1_classified3.append(i)
            testdata_classified3.append(i)

for i in testdata2:
    x=np.array(i).reshape(2,1)
    p1=discriminant_function(x,mean_class1,inv_covmat1,det_covmat1)
    p2=discriminant_function(x,mean_class2,inv_covmat2,det_covmat2)
    p3=discriminant_function(x,mean_class3,inv_covmat3,det_covmat3)
    if (p1-p2)>0:
        if (p1-p3)>0:
            class2_classified1.append(i)
            testdata_classified1.append(i)
        else:
#            if (p3-p1)>0 or p3==p1:
            class2_classified3.append(i)
            testdata_classified3.append(i)
    elif (p2-p1)>0 or p1==p2:
        if (p2-p3)>0:
            class2_classified2.append(i)
            testdata_classified2.append(i)
        else:
#            if (p3-p2)>0 or p3==p2:
            class2_classified3.append(i)
            testdata_classified3.append(i)

for i in testdata3:
    x=np.array(i).reshape(2,1)
    p1=discriminant_function(x,mean_class1,inv_covmat1,det_covmat1)
    p2=discriminant_function(x,mean_class2,inv_covmat2,det_covmat2)
    p3=discriminant_function(x,mean_class3,inv_covmat3,det_covmat3)
    if (p1-p2)>0:
        if (p1-p3)>0:
            class3_classified1.append(i)
            testdata_classified1.append(i)
        else:
#            if (p3-p1)>0 or p3==p1:
            class3_classified3.append(i)
            testdata_classified3.append(i)
    elif (p2-p1)>0 or p1==p2:
        if (p2-p3)>0:
            class3_classified2.append(i)
            testdata_classified2.append(i)
        else:
#            if (p3-p2)>0 or p3==p2:
            class3_classified3.append(i)
            testdata_classified3.append(i)

plt.scatter(*zip(*testdata_classified1),c="blue")
plt.scatter(*zip(*testdata_classified2),c="orange")
plt.scatter(*zip(*testdata_classified3),c="black")
plt.legend(['class1_train','class2_train','class3_train','class1_test','class2_test','class3_test'])
plt.title('Real world dataset')

#Green Blue-class1
#Red Orange-class2
#Magenta Black-class3

#Confusion matrix
matrix_for_number_of_elements = [[len(class1_classified1),len(class1_classified2),len(class1_classified3)],[len(class2_classified1),len(class2_classified2),len(class2_classified3)],[len(class3_classified1),len(class3_classified2),len(class3_classified3)]]
accuracy = (matrix_for_number_of_elements[0][0] + matrix_for_number_of_elements[1][1] + matrix_for_number_of_elements[2][2])*100/(matrix_for_number_of_elements[0][0] + matrix_for_number_of_elements[0][1] + matrix_for_number_of_elements[0][2] + matrix_for_number_of_elements[1][0] + matrix_for_number_of_elements[1][1] + matrix_for_number_of_elements[1][2] + matrix_for_number_of_elements[2][0] + matrix_for_number_of_elements[2][1] + matrix_for_number_of_elements[2][2])
print('Accuracy is ',accuracy,'%')
confusion_dict = {'':["class1","class2","class3"],'classified1':[len(class1_classified1),len(class2_classified1),len(class3_classified1)],'classified2':[len(class1_classified2),len(class2_classified2),len(class3_classified2)],'classified3':[len(class1_classified3),len(class2_classified3),len(class3_classified3)]}
# creating a dataframe from a dictionary
df = pd.DataFrame(confusion_dict)
print(df)