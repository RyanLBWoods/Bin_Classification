from sklearn.feature_selection import RFE
from sklearn import model_selection
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt

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

# Plot transformed X and Y
plt.figure(figsize=(15, 8))
ax1 = plt.subplot(221)
ax1.set_title("Transformed X and Y for Linear SVC")
ax1.set_xlabel("Wavelength")
ax1.set_ylabel("Spectrum")
ax1.set_xticks([500, 600, 700], ('500', '600', '700'))
y_list = y.tolist()
for i in range(len(y_list)):
    if y_list[i] == 0:
        ax1.scatter(wl[selected_index1], x.iloc[i, selected_index1], color='g', marker='o')
    else:
        ax1.scatter(wl[selected_index1], x.iloc[i, selected_index1], color='r', marker='o')


# Predict
predict = clf.predict(test)
print(predict)

ax2 = plt.subplot(222)
ax2.set_title("Predicted Result")
ax2.set_xlabel("Wavelength")
ax2.set_ylabel("Spectrum")
ax2.set_xticks([500, 600, 700], ('500', '600', '700'))

for i in range(len(predict)):
    if predict[i] == 0:
        ax2.scatter(wl[selected_index1], test.iloc[i, selected_index1], color='g', marker='x')
    else:
        ax2.scatter(wl[selected_index1], test.iloc[i, selected_index1], color='r', marker='x')

dt = DecisionTreeClassifier()
dt_selector = RFE(estimator=dt, n_features_to_select=1)
dt_selector.fit_transform(x, y)
dt_clf = DecisionTreeClassifier()
dt_clf.fit(x_train, y_train)

# Get index of selected feature
selected_index2 = int(dt_selector.get_support(True))

print("Feature importance ranking", dt_selector.ranking_)
print("Number of selected feature: %d" % dt_selector.n_features_)
print("The %dth was selected" % (selected_index2 + 1), "which is wavelength of %f" % wl[selected_index2])
print("Accuracy: %s" % dt_clf.score(x_test, y_test))

ax3 = plt.subplot(223)
ax3.set_title("Transformed X and Y for Decision Tree")
ax3.set_xlabel("Wavelength")
ax3.set_ylabel("Spectrum")
ax3.set_xticks([500, 600, 700], ('500', '600', '700'))
y_list = y.tolist()

for i in range(len(y_list)):
    if y_list[i] == 0:
        ax3.scatter(wl[selected_index2], x.iloc[i, selected_index2], color='g', marker='o')
    else:
        ax3.scatter(wl[selected_index2], x.iloc[i, selected_index2], color='r', marker='o')

# Predict
dt_predict = dt_clf.predict(test)
print(dt_predict)

ax4 = plt.subplot(224)
ax4.set_title("Predicted Result")
ax4.set_xlabel("Wavelength")
ax4.set_ylabel("Spectrum")
ax4.set_xticks([500, 600, 700], ('500', '600', '700'))

for i in range(len(dt_predict)):
    if dt_predict[i] == 0:
        ax4.scatter(wl[selected_index2], test.iloc[i, selected_index2], color='g', marker='x')
    else:
        ax4.scatter(wl[selected_index2], test.iloc[i, selected_index2], color='r', marker='x')

plt.tight_layout()
plt.show()
