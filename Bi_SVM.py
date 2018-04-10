import pandas as pd
import time
from sklearn import cross_validation, svm
import matplotlib.pyplot as plt

# Load data
x = pd.read_csv('./binary/X.csv', header=None)
y = pd.read_csv('./binary/Y.csv', header=None)
wl = pd.read_csv('./binary/Wavelength.csv', header=None)
x_test = pd.read_csv('./binary/XToClassify.csv', header=None)

# Convert y and wl to one-dimensional array
y = y[0]
wl = wl[0]

start = time.time()
clf = svm.SVC(kernel='linear')
# Cross validation
t_mae = - cross_validation.cross_val_score(clf, x, y, cv=5, scoring='neg_mean_absolute_error')
t_mse = - cross_validation.cross_val_score(clf, x, y, cv=5, scoring='neg_mean_squared_error')
print('Mean absolute error of 5-fold cross validation：', t_mae)
print('Mean squared error of 5-fold cross validation：',t_mse)
# Train model
start = time.time()
clf.fit(x, y)
print('Training cost', time.time() - start, 'second')
start = time.time()
y_predict = clf.predict(x_test)
print('Prediction test cost', time.time() - start, 'second')

# Output prediction to csv file
output = pd.DataFrame(y_predict)
output.to_csv('./binary/PredictedClass_SVMLinear.csv', index=None, header=None)

# Plot output
plt.figure(figsize=(15, 8))
ax1 = plt.subplot(111)
ax1.set_title("Predicted Color of Optical Reflectance Intensity of Spectrum")
ax1.grid(color='lightgrey', linestyle='-', linewidth=0.5)
plt.xlabel('Wavelength')
plt.ylabel('Optical Reflectance Intensity')
for i in range(len(y_predict)):
    if y_predict[i] == 0:
        plt.plot(wl, x_test.iloc[i], 'g', linewidth=0.5)
    else:
        plt.plot(wl, x_test.iloc[i], 'r', linewidth=0.5)
plt.tight_layout()
plt.show()