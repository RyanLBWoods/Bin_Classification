import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import svm

# Load data
x = pd.read_csv('./binary/X.csv', header=None)
y = pd.read_csv('./binary/Y.csv', header=None)
# Convert y to one-dimensional array
y = y[0]
# Split train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

print(x_train)
print(y_train)
# print(x_test)
# print(y_test)

clf = svm.SVC(kernel='rbf')
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)

print(y_predict)
print(y_test)