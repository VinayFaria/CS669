#=============================================================================
# Importing library
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state=0)

#=============================================================================
# StandardScaler() will normalize the features i.e. each column of X, INDIVIDUALLY, so that each column/feature/variable will have μ = 0 and σ = 1.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#=============================================================================
# Training the Algorithm
#Lets choose Gaussian kernel 
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

#=============================================================================
# Making Predictions
y_pred = classifier.predict(X_test)

#=============================================================================
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#=============================================================================
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM Classifier (Training set)')
plt.legend()
plt.show()

#=============================================================================
# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('green', 'red')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM Classifier (Test set)')
plt.legend()
plt.show()

"""
References
https://medium.com/pursuitnotes/day-12-kernel-svm-non-linear-svm-5fdefe77836c
"""