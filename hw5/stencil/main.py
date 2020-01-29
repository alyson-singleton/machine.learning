import numpy as np
import pandas as pd
from models import RegularizedLogisticRegression

def extract():
	X_train = pd.read_csv('../data/X_train.csv',header=None)
	y_train = pd.read_csv('../data/y_train.csv',header=None)
	X_val = pd.read_csv('../data/X_val.csv',header=None)
	y_val = pd.read_csv('../data/y_val.csv',header=None)

	y_train = np.array([i[0] for i in y_train.values])
	y_val = np.array([i[0] for i in y_val.values])

	X_train = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
	X_val = np.append(X_val, np.ones((len(X_val), 1)), axis=1)

	return X_train, X_val, y_train, y_val


def main():
	X_train, X_val, y_train, y_val = extract()
	RR = RegularizedLogisticRegression()
	RR.train(X_train, y_train)
	print('Train Accuracy: ' + str(RR.accuracy(X_train, y_train)))
	print('Validation Accuracy: ' + str(RR.accuracy(X_val, y_val)))
	#RR.plotError(X_train, y_train, X_val, y_val)

	#[TODO] Once implemented, call "plotError" here!

if __name__ == '__main__':
	main()
