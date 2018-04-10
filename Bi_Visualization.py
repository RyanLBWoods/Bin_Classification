import matplotlib.pyplot as plt
import pandas as pd

# Load data
x = pd.read_csv('./binary/X.csv', header=None)
y = pd.read_csv('./binary/Y.csv', header=None)
wl = pd.read_csv('./binary/Wavelength.csv', header=None)

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
        plt.plot(wl, x.iloc[i], 'g', linewidth=0.5)
    else:
        plt.plot(wl, x.iloc[i], 'r', linewidth=0.5)
plt.tight_layout()
plt.show()
