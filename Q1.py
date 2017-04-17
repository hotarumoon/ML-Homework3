import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt # import

# Load in the data with `read_csv()`
digits = pd.read_csv("ftp://ftp.ics.uci.edu/pub/ml-repos/machine-learning-databases/optdigits/optdigits.tra", header=None)
training_labels = digits.iloc[:,-1]
training_data = digits.iloc[:,:64]
X_train, X_test, y_train, y_test = train_test_split(training_data, training_labels, test_size = .1)

myList = list(range(1,50)) # creating odd list of K for KNN
neighbors = [x for x in myList if x % 2 != 0] # subsetting just the odd ones
cv_scores = [] # empty list that will hold cv scores

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

MSE = [1 - x for x in cv_scores] # changing to misclassification error
optimal_k = neighbors[MSE.index(min(MSE))]# determining best k
print("The optimal number of neighbors is: ")
print(optimal_k)

#KNN Classifier
knnClassifier = KNeighborsClassifier(optimal_k)
knnClassifier.fit(X_train,y_train)

predictions = knnClassifier.predict(X_test)
print('KNN ACCURACY:')
print(accuracy_score(y_test,predictions)) # evaluate accuracy

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()


#Decision Tree

decisionTreeClassifier = tree.DecisionTreeClassifier()
decisionTreeClassifier.fit(X_train,y_train)

predictions = decisionTreeClassifier.predict(X_test)
print('DecisionTreeClassifier ACCURACY:')
print(accuracy_score(y_test,predictions)) # evaluate accuracy
