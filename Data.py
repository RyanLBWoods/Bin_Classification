import pandas as pd

# Load data
x = pd.read_csv('./binary/X.csv', header=None)
y = pd.read_csv('./binary/Y.csv', header=None)

print(x.head())
print(y.head())
