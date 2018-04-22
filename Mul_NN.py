import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn import neural_network
from sklearn import model_selection

# Load data
x = pd.read_csv('./multiclass/X.csv', header=None)
y = pd.read_csv('./multiclass/Y.csv', header=None)
wl = pd.read_csv('./multiclass/Wavelength.csv', header=None)
test = pd.read_csv('./multiclass/XtoClassify.csv', header=None)

# Convert y and wl to one-dimensional array
y = y[0]
wl = wl[0]

# Split training and testing sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

hl = []
n = 1
best_hls = {}
while n < 11:
    for i in range(10):
        hl.append(n)
        hls = tuple(hl)
        clf = neural_network.MLPClassifier(hidden_layer_sizes=hls, solver='lbfgs', activation='relu', max_iter=1000, random_state=0)
        start = time.time()
        clf.fit(x_train, y_train)
        cost = time.time() - start
        y_predict = clf.predict(x_test)
        correct = 0
        for j in range(len(y_predict)):
            if y_predict[j] == y_test.iloc[j]:
                correct = correct + 1
        ac = correct / len(y_test)
        if ac == 1:
            best_hls[hls] = cost
    hl = []
    n = n + 1
fastest = min(zip(best_hls.values(), best_hls.keys()))
print(fastest)

# Classification
clf = neural_network.MLPClassifier(hidden_layer_sizes=fastest[1], solver='lbfgs', activation='relu', random_state=1)

# Train model
start = time.time()
clf.fit(x, y)
print('Training cost', time.time() - start, 'seconds')
y_predict = clf.predict(test)

# Output prediction to csv file
# output = pd.DataFrame(y_predict)
# output.to_csv('./multiclass/PredictedClass_NeuralNet.csv', index=None, header=None)

# Plot output
plt.figure(figsize=(15, 8))
ax1 = plt.subplot(111)
ax1.set_title("Predicted Color of Optical Reflectance Intensity of Spectrum Using Neural Net")
ax1.grid(color='lightgrey', linestyle='-', linewidth=0.5)
plt.xlabel('Wavelength')
plt.ylabel('Optical Reflectance Intensity')
for i in range(len(y_predict)):
    if y_predict[i] == 0:
        plt.plot(wl, test.iloc[i], 'b', linewidth=0.5)
    elif y_predict[i] == 1:
        plt.plot(wl, test.iloc[i], 'g', linewidth=0.5)
    elif y_predict[i] == 2:
        plt.plot(wl, test.iloc[i], 'pink', linewidth=0.5)
    elif y_predict[i] == 3:
        plt.plot(wl, test.iloc[i], 'r', linewidth=0.5)
    elif y_predict[i] == 4:
        plt.plot(wl, test.iloc[i], 'y', linewidth=0.5)
plt.tight_layout()
plt.show()
