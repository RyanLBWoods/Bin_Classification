import matplotlib.pyplot as plt
import pandas as pd

# Load data
x = pd.read_csv('./multiclass/X.csv', header=None)
y = pd.read_csv('./multiclass/Y.csv', header=None)
wl = pd.read_csv('./multiclass/Wavelength.csv', header=None)

# Convert y and wl to one-dimensional array
y = y[0]
wl = wl[0]

# Visualize data
plt.figure(figsize=(15, 8))
ax1 = plt.subplot(111)
ax1.set_title("Optical Reflectance Intensity of Spectrum")
ax1.grid(color='lightgrey', linestyle='-', linewidth=0.5)
plt.xlabel('Wavelength')
plt.ylabel('Optical Reflectance Intensity')

y = y.tolist()
for i in range(len(y)):
    if y[i] == 0:
        plt.plot(wl, x.iloc[i], 'b', linewidth=0.5)
    elif y[i] == 1:
        plt.plot(wl, x.iloc[i], 'g', linewidth=0.5)
    elif y[i] == 2:
        plt.plot(wl, x.iloc[i], 'pink', linewidth=0.5)
    elif y[i] == 3:
        plt.plot(wl, x.iloc[i], 'r', linewidth=0.5)
    elif y[i] == 4:
        plt.plot(wl, x.iloc[i], 'y', linewidth=0.5)
plt.tight_layout()
plt.show()
