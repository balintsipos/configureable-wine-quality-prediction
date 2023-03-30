import pandas as pd
from sklearn.svm import SVR

df = pd.read_csv('winequality.csv')
X = df.drop(columns='quality')
y = df['quality']

model = SVR()
model.fit(X, y)
predictions = model.predict(X)
print(predictions)
