import os
import argparse
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description='Train an SVR and get results')

parser.add_argument('-l', '--location', metavar='', type=str, help='Location of your dataset', default = './data/winequality.csv')
parser.add_argument('-tf', '--target_folder', metavar='', type=str, help='Target folder for results', default='./results')
parser.add_argument('-ts', '--test_split', metavar='', type=float, help='Size of the test set compared to the entire dataset', default=0.3)
parser.add_argument('-k', '--kernel', metavar='', type=str, help='The kernel you wish to train the SVM with', default='linear')
parser.add_argument('-d', '--degree', metavar='', type=int, help='Degree of the polynomial kernel function, must be non-negative', default=3)
parser.add_argument('-g', '--gamma', metavar='', type=str, help='Kernel coefficient for rbf, poly and sigmoid', default='scale')
parser.add_argument('-c0' ,'--coef0', metavar='', type=float, help='Independent term in kernel function. It is only significant in poly and sigmoid', default=0.0)
parser.add_argument('-t', '--tol', metavar='', type=float, help='Tolerance for stopping criterion', default=1e-3)
parser.add_argument('-C', '--C', metavar='', type=float, help='Regularization parameter. The strength of the regularization is inversely proportional to C', default=1.0)
parser.add_argument('-e', '--epsilon', metavar='', type=float, help='Epsilon in the epsilon-SVR model', default=0.1)
parser.add_argument('-s', '--shrinking', metavar='', type=bool, help='Whether to use the shrinking heuristic', default=True)
parser.add_argument('-cs', '--cache_size', metavar='', type=float, help='Specify the size of the kernel cache in MB', default=200)
parser.add_argument('-v' , '--verbose', metavar='', type=bool, help='Enable verbose output',default=False)
parser.add_argument('-mi', '--max_iter', metavar='', type=int, help='Hard limit on iterations within solver, or -1 for no limit', default=-1)

args = parser.parse_args()

df = pd.read_csv(args.location)
X = df.drop(columns='quality')
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_split)

# Making sure both float and 'scale'/'auto' types are supported
gamma = args.gamma
if isinstance(gamma, int) or isinstance(gamma, float):
    gamma = float(gamma)

model = SVR(
    kernel=args.kernel,
    degree=args.degree,
    gamma=gamma,
    coef0=args.coef0,
    tol=args.tol,
    C=args.C,
    epsilon=args.epsilon,
    shrinking=args.shrinking,
    cache_size=args.cache_size,
    verbose=args.verbose,
    max_iter=args.max_iter
    )

if not os.path.exists(args.target_folder):
    os.makedirs(args.target_folder)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Residual plot to visualize the magnitude of difference between predicted and actual
sns.residplot(x=y_pred, y=y_test)
sns.set_theme()
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual plot (R2: {:.2f}, MSE: {:.2f})'.format(r2, mse))
plt.savefig(os.path.join(args.target_folder, 'residualplot.png'))
plt.clf()

# Regression plot, the closer to the line, the better
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
sns.regplot(x='Actual', y='Predicted', data=df)
sns.set_theme()
plt.title('Regression plot for Actual vs Predicted values')
plt.savefig(os.path.join(args.target_folder, 'regplot.png'))
plt.clf()

# Histogram to see how each bin compares to the nr of actual values
plt.hist([y_test, y_pred], color= ['r', 'b'], bins=5)
plt.legend (['Actual', 'Predicted'], loc = 'upper right')
plt.title('Histogram of predicted wine quality')
plt.xlabel('Wine quality')
plt.ylabel('Frequency')
plt.savefig(os.path.join(args.target_folder, 'histplot.png'))

# No confusion matrix as this is a regression model