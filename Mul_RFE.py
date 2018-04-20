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

# Plot transformed X and Y
plt.figure(figsize=(15, 8))
ax1 = plt.subplot(221)
ax1.set_title("Transformed X and Y for Linear SVC")
ax1.set_xlabel("Optical Reflectance Intensity on Wavelength %f" % wl[selected_indexes1[0]])
ax1.set_ylabel("Optical Reflectance Intensity on Wavelength %f" % wl[selected_indexes1[1]])
ax1.set_xticks([500, 600, 700], ('500', '600', '700'))
y_list = y.tolist()
for i in range(len(y_list)):
    if y_list[i] == 0:
        ax1.scatter(x.iloc[i, selected_indexes1[0]], x.iloc[i, selected_indexes1[1]], color='b', marker='o')
    elif y_list[i] == 1:
        ax1.scatter(x.iloc[i, selected_indexes1[0]], x.iloc[i, selected_indexes1[1]], color='g', marker='o')
    elif y_list[i] == 2:
        ax1.scatter(x.iloc[i, selected_indexes1[0]], x.iloc[i, selected_indexes1[1]], color='pink', marker='o')
    elif y_list[i] == 3:
        ax1.scatter(x.iloc[i, selected_indexes1[0]], x.iloc[i, selected_indexes1[1]], color='r', marker='o')
    elif y_list[i] == 4:
        ax1.scatter(x.iloc[i, selected_indexes1[0]], x.iloc[i, selected_indexes1[1]], color='y', marker='o')

# Predict
predict = clf.predict(test)
print(predict)

ax2 = plt.subplot(222)
ax2.set_title("Predict Result")
ax2.set_xlabel("Optical Reflectance Intensity on Wavelength %f" % wl[selected_indexes1[0]])
ax2.set_ylabel("Optical Reflectance Intensity on Wavelength %f" % wl[selected_indexes1[1]])
ax2.set_xticks([500, 600, 700], ('500', '600', '700'))
for i in range(len(predict)):
    if predict[i] == 0:
        ax2.scatter(test.iloc[i, selected_indexes1[0]], test.iloc[i, selected_indexes1[1]], color='b', marker='x')
    elif predict[i] == 1:
        ax2.scatter(test.iloc[i, selected_indexes1[0]], test.iloc[i, selected_indexes1[1]], color='g', marker='x')
    elif predict[i] == 2:
        ax2.scatter(test.iloc[i, selected_indexes1[0]], test.iloc[i, selected_indexes1[1]], color='pink', marker='x')
    elif predict[i] == 3:
        ax2.scatter(test.iloc[i, selected_indexes1[0]], test.iloc[i, selected_indexes1[1]], color='r', marker='x')
    elif predict[i] == 4:
        ax2.scatter(test.iloc[i, selected_indexes1[0]], test.iloc[i, selected_indexes1[1]], color='y', marker='x')


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
wl_list = []
for index in selected_indexes2:
    print(wl[index], end=" ")
    wl_list.append(wl[index])
print("]")
print("Accuracy: %s" % dt_clf.score(x_test, y_test))

ax3 = plt.subplot(223)
ax3.set_title("Transformed X and Y for Decision Tree")
ax3.set_xlabel("Selected Wavelength")
ax3.set_ylabel("Optical Reflectance Intensity on Wavelength")
ax3.set_xticks([500, 600, 700], ('500', '600', '700'))
for i in range(len(y_list)):
    if y_list[i] == 0:
        ax3.plot(wl_list, x.iloc[i, selected_indexes2], color='b')
    elif y_list[i] == 1:
        ax3.plot(wl_list, x.iloc[i, selected_indexes2], color='g')
    elif y_list[i] == 2:
        ax3.plot(wl_list, x.iloc[i, selected_indexes2], color='pink')
    elif y_list[i] == 3:
        ax3.plot(wl_list, x.iloc[i, selected_indexes2], color='r')
    elif y_list[i] == 4:
        ax3.plot(wl_list, x.iloc[i, selected_indexes2], color='y')

# Predict
dt_predict = dt_clf.predict(test)
print(dt_predict)

ax4 = plt.subplot(224)
ax4.set_title("Predict Result")
ax4.set_xlabel("Selected Wavelength")
ax4.set_ylabel("Optical Reflectance Intensity on Wavelength")
ax4.set_xticks([500, 600, 700], ('500', '600', '700'))
for i in range(len(dt_predict)):
    if dt_predict[i] == 0:
        ax4.plot(wl_list, test.iloc[i, selected_indexes2], color='b')
    elif dt_predict[i] == 1:
        ax4.plot(wl_list, test.iloc[i, selected_indexes2], color='g')
    elif dt_predict[i] == 2:
        ax4.plot(wl_list, test.iloc[i, selected_indexes2], color='pink')
    elif dt_predict[i] == 3:
        ax4.plot(wl_list, test.iloc[i, selected_indexes2], color='r')
    elif dt_predict[i] == 4:
        ax4.plot(wl_list, test.iloc[i, selected_indexes2], color='y')

plt.tight_layout()
plt.show()
