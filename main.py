import pandas as pd
import sys
import argparse
from sklearn.svm import SVR

parser = argparse.ArgumentParser(description='prints the arg to cli')

parser.add_argument('-message', metavar='message', type=str, help='enter your message')
parser.add_argument('--kernel', metavar='kernel', type=str, help='the kernel you wish to train the SVM with', default='linear')
parser.add_argument('--degree', metavar='degree', type=int, help='Degree of the polynomial kernel function, must be non-negative', default=3)
parser.add_argument('--gamma', metavar='gamma', type=str, help='Kernel coefficient for rbf, poly and sigmoid', default=3)
parser.add_argument('--coef0', metavar='coef0', type=float, help='Independent term in kernel function. It is only significant in poly and sigmoid', default=0.0)


args = parser.parse_args()
message = args.message


df = pd.read_csv('winequality.csv')
X = df.drop(columns='quality')
y = df['quality']

model = SVR(
    kernel=args.kernel,
    degree=args.degree,
    gamma=args.gamma,
    coef0=args.coef0
    )


model.fit(X, y)
predictions = model.predict(X)
print(predictions)
print(type(message))
