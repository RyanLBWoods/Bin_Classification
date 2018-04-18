from sklearn.feature_selection import RFE
from sklearn import model_selection
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt

# Load data
x = pd.read_csv('./multiclass/X.csv', header=None)
y = pd.read_csv('./multiclass/Y.csv', header=None)
wl = pd.read_csv('./multiclass/Wavelength.csv', header=None)
test = pd.read_csv('./multiclass/XToClassify.csv', header=None)

# Convert y and wl to one-dimensional array
y = y[0]
wl = wl[0]
wl = wl.values
wl = wl.tolist()
# Name columns with wavelength
x.columns = wl
test.columns = wl
# Remove features that are obvious less important
for col in x.columns:
    if col < 550 or col > 650:
        del x[col]
        del test[col]
wl = x.columns  # Get remained wavelength
wl = wl.tolist()

# Split training and testing sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

# Select features and fit the model
lsvc = LinearSVC()
svc_selector = RFE(estimator=lsvc, n_features_to_select=2)
svc_selector.fit_transform(x, y)
clf = LinearSVC()
clf.fit(x_train, y_train)

# Get index of selected feature
selected_indexes1 = svc_selector.get_support(True)
for value in selected_indexes1:
    value = int(value)

print(svc_selector.ranking_)
print("Number of selected feature: %d" % svc_selector.n_features_)
print("The %dth and % dth was selected" % ((selected_indexes1[0] + 1), (selected_indexes1[1] + 1)),
      "which is wavelength of %f and %f" % (wl[selected_indexes1[0]], wl[selected_indexes1[1]]))
print("Accuracy: %s" % clf.score(x_test, y_test))

# Predict
predict = clf.predict(test)
print(predict)

# Select features and fit the model
dt = DecisionTreeClassifier()
dt_selector = RFE(estimator=dt, n_features_to_select=110)
dt_selector.fit_transform(x, y)
dt_clf = DecisionTreeClassifier()
dt_clf.fit(x_train, y_train)

# Get index of selected feature
selected_indexes2 = dt_selector.get_support(True)
for value in selected_indexes2:
    value = int(value)

print(dt_selector.ranking_)
print("Number of selected feature: %d" % dt_selector.n_features_)
print("The selected indexes of features are", selected_indexes2)
print("They are wavelength of [", end="")
for index in selected_indexes2:
    print(wl[index], end=" ")
print("]")
print("Accuracy: %s" % dt_clf.score(x_test, y_test))

# Predict
dt_predict = dt_clf.predict(test)
print(dt_predict)
