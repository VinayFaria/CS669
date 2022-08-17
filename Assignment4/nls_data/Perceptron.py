#============================================================================
# Importing library
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import time

#============================================================================
tic=time.perf_counter()
dimension = 2
splitRatio = 0.70 #70% training data & 30% testing data

#============================================================================
# Data Preprocessing
labels = ['Class1', 'Class2']
colors = ['red', 'green', 'yellow']
column_Names=["axis1","axis2"]
Class1_data = pd.read_csv("class1.txt", names = column_Names)
Class2_data = pd.read_csv("class2.txt", names = column_Names)
# class1 is taken as +ve
Class1_label = np.zeros(shape=(Class1_data.shape[0]), dtype = int)
Class1_label.fill(1)
# class2 is taken as -ve
Class2_label = np.zeros(shape=(Class2_data.shape[0]), dtype = int)
Class2_label.fill(-1)

Data =  np.concatenate( (Class1_data, Class2_data),axis = 0)

x0 = np.zeros((Data.shape[0],1),dtype = float)
x0.fill(1)
Data = np.hstack((x0,Data))

Total_samples = int(Data.shape[0])
Train_samples = int (splitRatio*Total_samples)
Test_samples = int((1-splitRatio)*Total_samples)

label = np.concatenate((Class1_label, Class2_label), axis =0)
Train_Data, Test_Data, Train_label, Test_label = train_test_split(Data,label,test_size=0.3)

weights = np.random.rand(dimension+1)

#============================================================================
def output(data, weight):
    if np.matmul(np.transpose(weight), data ) > 0:
        return 1
    elif np.matmul(np.transpose(weight), data ) < 0:
        return -1
    elif np.matmul(np.transpose(weight), data ) == 0:
        return 0

#============================================================================
# Training weights
Predicted_Class1 = []
Predicted_Class2 = []
misclassified = True
epoch_index = 0
EpochWise_Loss = [[],[]]
while misclassified == True:
    index = 0
    if epoch_index == 30:
        break
    wrong_Classified = 0
    misclassified = False
    Predicted_Class1.clear()
    Predicted_Class2.clear()
    for sample in Train_Data:
        y = output(sample, weights)
        if y != Train_label[index]:
            wrong_Classified += 1
            misclassified = True
            weights = weights +(-y)*(sample)
        if y == 1:
            Predicted_Class1.append(sample)
        else:
            Predicted_Class2.append(sample)
        index += 1

    Loss = (wrong_Classified / float(Train_samples))*100
    EpochWise_Loss[0].append(epoch_index)
    EpochWise_Loss[1].append(Loss)
    epoch_index += 1

Predicted_Class1 = np.asarray(Predicted_Class1)
Predicted_Class2 = np.asarray(Predicted_Class2)

Predicted_Class1 = Predicted_Class1[:,1:]
Predicted_Class2 = Predicted_Class2[:,1:]

#============================================================================
# Plotting Training data
plt.figure(1)
plt.scatter(Predicted_Class1[:,0], Predicted_Class1[:,1],c='red',s=10,label="class1")
plt.scatter(Predicted_Class2[:,0], Predicted_Class2[:,1],c='green',s=10,label="class2")

x = np.linspace(min(min(Predicted_Class1[:,0]),min(Predicted_Class2[:,0])), max(max(Predicted_Class1[:,0]),max(Predicted_Class2[:,0])),100)
y = -(weights[1]*x)/weights[2] - weights[0]/weights[2]  #y=mx+c
#ref https://www.thomascountz.com/2018/04/13/calculate-decision-boundary-of-perceptron

slope = -((weights[1])/weights[2])  #m
intercept = - (weights[0]/(weights[2])) #c
plt.plot(x,y,'-r',label='y = %.2fx + %.2f' %(slope, intercept))
plt.legend()
plt.title("Training Data")
plt.savefig("output_plot/Training_Data.jpg")
plt.show()

#============================================================================
# Plotting Training Decision Boundary
class1_train_data =[]
class2_train_data =[]
for i in range(len(Train_Data)):
	x=Train_Data[i]
	arg1 = x[0]*weights[0]+x[1]*weights[1]+x[2]*weights[2]
	if arg1 > 0:
		class1_train_data.append(x)
	else:
		class2_train_data.append(x)

Decision_boundary1 = []
Decision_boundary2 = []

for x in np.arange(min(min(Predicted_Class1[:,0]),min(Predicted_Class2[:,0])), max(max(Predicted_Class1[:,0]),max(Predicted_Class2[:,0])),0.05):
	for y in np.arange(-30, 40, 0.05):
		arg1 = 1*weights[0]+x*weights[1]+y*weights[2]

		if arg1 > 0:
			Decision_boundary1.append([1, x, y])
		else:
			Decision_boundary2.append([1, x, y])

A=[]
B=[]
for i in Decision_boundary1:
	A.append(i[1])
	B.append(i[2])
plt.figure(2)
plt.scatter(A, B, c='Grey', alpha=0.5 )
A=[]
B=[]
for i in Decision_boundary2:
	A.append(i[1])
	B.append(i[2])
plt.scatter(A, B, c='pink', alpha=0.5)

A=[]
B=[]
for i in class1_train_data:
	A.append(i[1])
	B.append(i[2])
plt.scatter(A,B,c='red',s=10,label=labels[0])

A=[]
B=[]
for i in class2_train_data:
	A.append(i[1])
	B.append(i[2])
plt.scatter(A,B,c='green',s=10,label=labels[1])
plt.title("Training Samples Decision Boundary")
plt.legend()
plt.savefig("output_plot/Train_Decision_Boundary.jpg")
plt.show()

#============================================================================
# Testing the splitted data
Predicted_Class1 = []
Predicted_Class2 = []
misclassified = True
epoch = 0
Y_pred = []
for sample in Test_Data:
    y = output(sample, weights)
    if y >0 :
        Predicted_Class1.append(sample)
        Y_pred.append(1)
    else:
        Predicted_Class2.append(sample)
        Y_pred.append(-1)
    index += 1
Y_pred = np.array(Y_pred)
conf_mat=confusion_matrix(Y_pred,Test_label)

Predicted_Class1 = np.asarray(Predicted_Class1)
Predicted_Class2 = np.asarray(Predicted_Class2)

Predicted_Class1 = Predicted_Class1[:,1:]

Predicted_Class2 = Predicted_Class2[:,1:]

plt.figure(3)
plt.plot(Predicted_Class1[:,0], Predicted_Class1[:,1],"o ", color=colors[0], label="class1")
plt.plot(Predicted_Class2[:,0], Predicted_Class2[:,1],"o", color=colors[1], label="class2")

x = np.linspace(min(min(Predicted_Class1[:,0]),min(Predicted_Class2[:,0])), max(max(Predicted_Class1[:,0]),max(Predicted_Class2[:,0])),100)
y = -(weights[1]*x)/weights[2] - weights[0]/weights[2]
slope = -((weights[1])/weights[2])
intercept = - (weights[0]/(weights[2]))
plt.plot(x,y,'-r',label='y = %.2fx + %.2f' %(slope, intercept))
plt.legend()
plt.title("Testing Data")
plt.savefig("output_plot/Testing_Data.jpg")
plt.show()

#============================================================================
# Plotting Testing Decision Boundary
class1_test_data =[]
class2_test_data =[]
for i in range(len(Test_Data)):
	x=Test_Data[i]
	arg1 = x[0]*weights[0]+x[1]*weights[1]+x[2]*weights[2]
	if arg1 > 0:
		class1_test_data.append(x)
	else:
		class2_test_data.append(x)
        
Decision_boundary1 = []
Decision_boundary2 = []

for x in np.arange(min(min(Predicted_Class1[:,0]),min(Predicted_Class2[:,0])), max(max(Predicted_Class1[:,0]),max(Predicted_Class2[:,0])),0.05):
	for y in np.arange(-30, 40, 0.05):
		arg1 = 1*weights[0]+x*weights[1]+y*weights[2]

		if arg1 > 0:
			Decision_boundary1.append([1, x, y])
		else:
			Decision_boundary2.append([1, x, y])

A=[]
B=[]
for i in Decision_boundary1:
	A.append(i[1])
	B.append(i[2])
plt.figure(4)
plt.scatter(A, B, c='Grey', alpha=0.5 )

A=[]
B=[]
for i in Decision_boundary2:
	A.append(i[1])
	B.append(i[2])
plt.scatter(A, B, c='pink', alpha=0.5)

class1_classified1_col1 = []
class1_classified1_col2 = []
class1_classified2_col1 = []
class1_classified2_col2 = []
class2_classified1_col1 = []
class2_classified1_col2 = []
class2_classified2_col1 = []
class2_classified2_col2 = []
dummy1 = []
dummy2 = []
for i in Test_Data:
    dummy1.append(i[1])
    dummy2.append(i[2])
for i,j,k,l in zip(dummy1, dummy2, Y_pred, Test_label):
    if k == l and k == 1:
        class1_classified1_col1.append(i)
        class1_classified1_col2.append(j)
    elif k != l and k == -1:
        class1_classified2_col1.append(i)
        class1_classified2_col2.append(j)
    elif k != l and k == 1:
        class2_classified1_col1.append(i)
        class2_classified1_col2.append(j)
    elif k == l and k == -1:
        class2_classified2_col1.append(i)
        class2_classified2_col2.append(j)

plt.scatter(class1_classified1_col1, class1_classified1_col2,c='purple',s=10)
plt.scatter(class1_classified2_col1, class1_classified2_col2,c='grey',s=10)
plt.scatter(class2_classified1_col1, class2_classified1_col2,c='gold',s=10)
plt.scatter(class2_classified2_col1, class2_classified2_col2,c='cyan',s=10)
plt.legend(['class1_train','class2_train','class1_classified1','class1_classified2','class2_classified1','class2_classified2'])
plt.title("Testing Samples Decision Boundary ")
plt.savefig("output_plot/Test_DecisionBoundary.jpg")
toc = time.perf_counter()

#============================================================================
print('-----------------------------------------------')
for i in range(len(EpochWise_Loss[0])):
    print("Epoch : ",i," Error :",EpochWise_Loss[1][i])
print('-----------------------------------------------')
print('confusion matrix')
print('-----------------------------------------------')
print(conf_mat)
print('-----------------------------------------------')
print(f'Time taken: {toc - tic:0.4f} seconds')  # printing the execution time
print('-----------------------------------------------')