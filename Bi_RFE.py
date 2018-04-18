from sklearn.feature_selection import RFE
from sklearn import model_selection
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Load data
x = pd.read_csv('./binary/X.csv', header=None)
y = pd.read_csv('./binary/Y.csv', header=None)
wl = pd.read_csv('./binary/Wavelength.csv', header=None)
test = pd.read_csv('./binary/XToClassify.csv', header=None)

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
    if col < 500 or col > 700:
        del x[col]
        del test[col]
wl = x.columns  # Get remained wavelength
wl = wl.tolist()

# Split training and testing sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

# Select feature and fit the model
lsvc = LinearSVC()
svc_selector = RFE(estimator=lsvc, n_features_to_select=1)
svc_selector.fit_transform(x, y)
clf = LinearSVC()
clf.fit(x_train, y_train)

# Get index of selected feature
selected_index1 = int(svc_selector.get_support(True))

print("Feature importance ranking", svc_selector.ranking_)
print("Number of selected feature: %d" % svc_selector.n_features_)
print("The %dth was selected" % (selected_index1 + 1), "which is wavelength of %f" % wl[selected_index1])
print("Accuracy: %s" % clf.score(x_test, y_test))

# Predict
predict = clf.predict(test)
print(predict)

dt = DecisionTreeClassifier()
dt_selector = RFE(estimator=dt, n_features_to_select=1)
dt_selector.fit_transform(x, y)
dt_clf = DecisionTreeClassifier()
dt_clf.fit(x_train, y_train)

# Get index of selected feature
selected_index2 = int(dt_selector.get_support(True))

print("Feature importance ranking", svc_selector.ranking_)
print("Number of selected feature: %d" % dt_selector.n_features_)
print("The %dth was selected" % (selected_index2 + 1), "which is wavelength of %f" % wl[selected_index2])
print("Accuracy: %s" % dt_clf.score(x_test, y_test))

# Predict
dt_predict = dt_clf.predict(test)
print(dt_predict)
