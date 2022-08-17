#=============================================================================
# Importing library
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#=============================================================================
# Data Preprocessing
col_Names=["axis1","axis2"]
class1 = pd.read_csv("class1.txt", names = col_Names)
class1["class"] = "1"
class1.to_csv("class1_update.txt", index=False)

class2 = pd.read_csv("class2.txt", names = col_Names)
class2["class"] = "2"
class2.to_csv("class2_update.txt", index=False)

interesting_files = ['class1_update.txt','class2_update.txt']
class1_class2 = pd.concat((pd.read_csv(f, header = 0) for f in interesting_files))
#class1_class2.to_csv("class1_class2.txt", index=False)

X = class1_class2.drop(['class'], axis=1)
Y = class1_class2['class']

#=============================================================================
# splitting training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30)
Y_train = Y_train.to_frame('class')

#=============================================================================
#Plotting the train data class1 & class2
class1_col1 = []
class1_col2 = []
class2_col1 = []
class2_col2 = []
dummy1 = X_train["axis1"].tolist()
dummy2 = X_train["axis2"].tolist()
dummy3 = Y_train["class"].tolist()
for i,j,k in zip(dummy1, dummy2, dummy3):
    if k == 1:
        class1_col1.append(i)
        class1_col2.append(j)
    elif k == 2:
        class2_col1.append(i)
        class2_col2.append(j)

plt.scatter(class1_col1, class1_col2,c='purple')
plt.scatter(class2_col1, class2_col2,c='cyan')

#=============================================================================
# Training the Algorithm
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, Y_train.values.ravel())

#=============================================================================
# Making Predictions
Y_pred = svclassifier.predict(X_test)

#=============================================================================
# Plotting the test data class1 & class2
class1_classified1_col1 = []
class1_classified1_col2 = []
class1_classified2_col1 = []
class1_classified2_col2 = []
class2_classified1_col1 = []
class2_classified1_col2 = []
class2_classified2_col1 = []
class2_classified2_col2 = []
dummy1 = X_test["axis1"].tolist()
dummy2 = X_test["axis2"].tolist()
for i,j,k,l in zip(dummy1, dummy2, Y_pred, Y_test):
    if k == l and k == 1:
        class1_classified1_col1.append(i)
        class1_classified1_col2.append(j)
    elif k != l and k == 2:
        class1_classified2_col1.append(i)
        class1_classified2_col2.append(j)
    elif k != l and k == 1:
        class2_classified1_col1.append(i)
        class2_classified1_col2.append(j)
    elif k == l and k == 2:
        class2_classified2_col1.append(i)
        class2_classified2_col2.append(j)

plt.scatter(class1_classified1_col1, class1_classified1_col2,c='brown')
plt.scatter(class1_classified2_col1, class1_classified2_col2,c='gray')
plt.scatter(class2_classified1_col1, class2_classified1_col2,c='pink')
plt.scatter(class2_classified2_col1, class2_classified2_col2,c='orange')

# Encircle support vectors
plt.scatter(svclassifier.support_vectors_[:, 0], svclassifier.support_vectors_[:, 1], s=50, facecolors='none', edgecolors='k', alpha=.5);
plt.legend(['class1_train','class2_train','class1_classified1','class1_classified2','class2_classified1','class2_classified2','support vector'])
plt.title('Lineraly separable dataset')

#=============================================================================
# Constructing a hyperplane using a formula.
w = svclassifier.coef_[0]           # w consists of 2 elements
b = svclassifier.intercept_[0]      # b consists of 1 element
x_points = np.linspace(min(min(class1_col1),min(class2_col1)), max(max(class1_col1),max(class2_col1)))    # generating x-points from -1 to 1
y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points
# Plotting a hyperplane
plt.plot(x_points, y_points, c='r');

#=============================================================================
# Plotting SVM margins

# Step 1 (unit-vector):
w_hat = svclassifier.coef_[0] / (np.sqrt(np.sum(svclassifier.coef_[0] ** 2)))
# Step 2 (margin):
margin = 1 / np.sqrt(np.sum(svclassifier.coef_[0] ** 2))
# Step 3 (calculate points of the margin lines):
decision_boundary_points = np.array(list(zip(x_points, y_points)))
points_of_line_above = decision_boundary_points + w_hat * margin
points_of_line_below = decision_boundary_points - w_hat * margin

# Plot margin lines
# Blue margin line above
plt.plot(points_of_line_above[:, 0], points_of_line_above[:, 1], 'b--', linewidth=2)
# Green margin line below
plt.plot(points_of_line_below[:, 0], points_of_line_below[:, 1], 'g--', linewidth=2)

#=============================================================================
# Evaluating the Algorithm
print(confusion_matrix(Y_test,Y_pred))

"""
References
https://muthu.co/understanding-support-vector-machines-using-python/
https://medium.com/geekculture/svm-classification-with-sklearn-svm-svc-how-to-plot-a-decision-boundary-with-margins-in-2d-space-7232cb3962c0
"""