#=============================================================================
# Importing library
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

#=============================================================================
def buildMLPerceptron(train_features, test_features, train_targets, test_targets, num_neurons=1):
    # This model optimizes the log-loss function using stochastic gradient descent.
    # The ith element represents the number of neurons in the ith hidden layer in the tuple
    # Activation function for the hidden layer.
    # The solver for weight optimization.
    # Learning rate schedule for weight updates.
    # Verbose Whether to print progress messages to stdout.
    # The solver iterates until convergence (determined by ‘tol’ , default=1e-4) or number of iterations (max default iteration = 200)
    classifier = MLPClassifier(hidden_layer_sizes=(num_neurons,), activation='relu', solver='sgd', learning_rate='invscaling', random_state=67, verbose=10)
    
    # Fit the model to data matrix 'train_features' and target(s) 'train_targets'.
    # Returns a trained MLP model.
    classifier.fit(train_features, train_targets.values.ravel())
    
    # Predict using the trained multi-layer perceptron classifier.
    # Returns the predicted classes.
    predictions = classifier.predict(test_features)
    
    # Accuracy classification score.
    score = np.round(metrics.accuracy_score(test_targets, predictions),4)
    print('Mean accurcy of predictons: '+str(score*100)+'%')
    return predictions

#=============================================================================
# Data Preprocessing
col_Names=["axis1","axis2"]
class1 = pd.read_csv("class1.txt", names = col_Names)
class1["class"] = "1"
#class1.to_csv("class1_update.txt", index=False)

class2 = pd.read_csv("class2.txt", names = col_Names)
class2["class"] = "2"
#class2.to_csv("class2_update.txt", index=False)

interesting_files = ['class1_update.txt','class2_update.txt']
class1_class2 = pd.concat((pd.read_csv(f, header = 0) for f in interesting_files))
#class1_class2.to_csv("class1_class2.txt", index=False)

X = class1_class2.drop(['class'], axis=1)
Y = class1_class2['class']

#=============================================================================
# splitting training and test
train_features, test_features, train_targets, test_targets = train_test_split(X, Y, test_size=0.3, random_state=123)
train_targets = train_targets.to_frame('class')

#=============================================================================
#Plotting the train data class1 & class2
class1_col1 = []
class1_col2 = []
class2_col1 = []
class2_col2 = []
dummy1 = train_features["axis1"].tolist()
dummy2 = train_features["axis2"].tolist()
dummy3 = train_targets["class"].tolist()
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
predicted = buildMLPerceptron(train_features, test_features, train_targets, test_targets)

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
dummy1 = test_features["axis1"].tolist()
dummy2 = test_features["axis2"].tolist()
for i,j,k,l in zip(dummy1, dummy2, predicted, test_targets):
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
plt.legend(['class1_train','class2_train','class1_classified1','class1_classified2','class2_classified1','class2_classified2'])
plt.title('Lineraly separable dataset')

"""
References
https://towardsdatascience.com/beginners-ask-how-many-hidden-layers-neurons-to-use-in-artificial-neural-networks-51466afa0d3e
https://towardsdatascience.com/multilayer-perceptron-explained-with-a-real-life-example-and-python-code-sentiment-analysis-cb408ee93141
"""