import pandas as pd
from sklearn import cross_validation, svm
import EvaluationFunctions

# Load data
x = pd.read_csv('./binary/X.csv', header=None)
y = pd.read_csv('./binary/Y.csv', header=None)
wl = pd.read_csv('./binary/Wavelength.csv', header=None)

# Convert y and wl to one-dimensional array
y = y[0]
wl = wl[0]
# Split train and test set
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y)

clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)

mae = EvaluationFunctions.mae(y_test, y_predict)
rmse = EvaluationFunctions.rmse(y_test, y_predict)

print('Mean Absolute Error: ', mae)
print('Rooted Mean Square Error: ', rmse)
