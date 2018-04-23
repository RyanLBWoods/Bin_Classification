from sklearn.feature_selection import RFECV
from sklearn import model_selection
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt

# Load data
x = pd.read_csv('./binary/X.csv', header=None)
y = pd.read_csv('./binary/y.csv', header=None)
wl = pd.read_csv('./binary/Wavelength.csv', header=None)
test = pd.read_csv('./binary/XToClassify.csv', header=None)

# Convert y and wl to one-dimensional array
y = y[0]
wl = wl[0]
# Name columns with wavelength
x.columns = wl.values
# Remove features that are obvious less important
for col in x.columns:
    if col < 500 or col > 700:
        del x[col]
# Split training and testing sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)
# RFECV with Linear SVC
lsvc = LinearSVC()
svc_selector = RFECV(estimator=lsvc, step=1, cv=model_selection.StratifiedKFold(3), scoring='accuracy')
svc_selector.fit(x, y)
print("Optimal number of features: %d" % svc_selector.n_features_)
print("Ranking of features: %s" % svc_selector.ranking_)

# Plot
plt.figure(figsize=(15, 8))
ax1 = plt.subplot(211)
ax1.set_title("RFECV with Linear SVC model")
ax1.set_xlabel("number of features selected")
ax1.set_ylabel("CV score")
ax1.plot(range(1, len(svc_selector.grid_scores_) + 1), svc_selector.grid_scores_)

# RFECV with Decision Tree Classifier
dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt_selector = RFECV(estimator=dt, step=1, cv=model_selection.StratifiedKFold(3), scoring='accuracy')
dt_selector.fit(x, y)
print("Optimal number of features: %d" % dt_selector.n_features_)
print("Ranking of features: %s" % dt_selector.ranking_)

# Plot
ax2 = plt.subplot(212)
ax2.set_title("RFECV with DecisionTreeClassifier model")
ax2.set_xlabel("number of features selected")
ax2.set_ylabel("CV score")
ax2.plot(range(1, len(dt_selector.grid_scores_) + 1), dt_selector.grid_scores_)

plt.tight_layout()
plt.show()
