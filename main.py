import pandas as pd
import argparse
from sklearn.svm import SVR

parser = argparse.ArgumentParser(description='prints the arg to cli')

parser.add_argument('-location', metavar='location', type=str, help='location of your dataset')
parser.add_argument('-message', metavar='message', type=str,help='enter your message')
parser.add_argument('--kernel', metavar='kernel', type=str, help='the kernel you wish to train the SVM with', default='linear')
parser.add_argument('--degree', metavar='degree', type=int, help='Degree of the polynomial kernel function, must be non-negative', default=3)
parser.add_argument('--gamma', metavar='gamma', type=str, help='Kernel coefficient for rbf, poly and sigmoid', default='scale')
parser.add_argument('--coef0', metavar='coef0', type=float, help='Independent term in kernel function. It is only significant in poly and sigmoid', default=0.0)
parser.add_argument('--tol', metavar='tol', type=float, help='Tolerance for stopping criterion', default=1e-3)
parser.add_argument('--C', metavar='C', type=float, help='Regularization parameter. The strength of the regularization is inversely proportional to C', default=1.0)
parser.add_argument('--epsilon', metavar='epsilon', type=float, help='Epsilon in the epsilon-SVR model', default=0.1)
parser.add_argument('--shrinking', metavar='shrinking', type=bool, help='Whether to use the shrinking heuristic', default=True)
parser.add_argument('--cache_size', metavar='cache-size', type=float, help='Specify the size of the kernel cache in MB', default=200)
parser.add_argument('--verbose', metavar='verbose', type=bool, help='Enable verbose output',default=False)
parser.add_argument('--max_iter', metavar='max_iter', type=int, help='Hard limit on iterations within solver, or -1 for no limit', default=-1)

args = parser.parse_args()
message = args.message


df = pd.read_csv(args.location)
X = df.drop(columns='quality')
y = df['quality']


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


model.fit(X, y)
predictions = model.predict(X)
print(predictions)
print(type(message))
