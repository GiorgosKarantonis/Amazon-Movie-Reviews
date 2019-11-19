import os
import subprocess
import sys
import numpy as np

from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN



def load_data_from_gcloud():
	X_train_file = os.path.join(os.getcwd(), 'X_train_file.csv')
	subprocess.check_call(['gsutil', 'cp', 'gs://cs-506-258209-mlengine/X_train_file.csv', X_train_file], stderr=sys.stdout)

	y_train_file = os.path.join(os.getcwd(), 'y_train_file.csv')
	subprocess.check_call(['gsutil', 'cp', 'gs://cs-506-258209-mlengine/y_train_file.csv', y_train_file], stderr=sys.stdout)

	y_train_one_hot_file = os.path.join(os.getcwd(), 'y_train_one_hot_file.csv')
	subprocess.check_call(['gsutil', 'cp', 'gs://cs-506-258209-mlengine/y_train_one_hot_file.csv', y_train_one_hot_file], stderr=sys.stdout)

	X_predict_file = os.path.join(os.getcwd(), 'X_predict_file.csv')
	subprocess.check_call(['gsutil', 'cp', 'gs://cs-506-258209-mlengine/X_predict_file.csv', X_predict_file], stderr=sys.stdout)

	X_train = np.genfromtxt(X_train_file, delimiter=',')
	y_train = np.genfromtxt(y_train_file, delimiter=',')
	y_train_one_hot = np.genfromtxt(y_train_one_hot_file, delimiter=',')
	X_predict = np.genfromtxt(X_predict_file, delimiter=',')

	return X_train, y_train, y_train_one_hot, X_predict


def load_data():
	X_train = np.genfromtxt('X_train.csv', delimiter=',')
	y_train = np.genfromtxt('./y_train.csv', delimiter=',')
	y_train_one_hot = np.genfromtxt('./y_train_one_hot.csv', delimiter=',')
	X_predict = np.genfromtxt('./X_predict.csv', delimiter=',')

	return X_train, y_train, y_train_one_hot, X_predict


def shift_data(X):
	return X + np.abs(np.min(X))



def logistic_regression(X, y, X_predict, solver='lbfgs'):
	model = LogisticRegression(random_state=0, solver=solver, multi_class='multinomial').fit(X, y)

	training_score = model.score(X, y)
	print('Logistic Regression: ', training_score)

	return model.predict(X_predict)


def svc(X, y, X_predict, kernel='linear', C=5):
	model = SVC(gamma='auto', C=C, kernel=kernel).fit(X, y)

	training_score = model.score(X, y)
	print('SVC: ', training_score)

	return model.predict(X_predict)


def decision_trees(X, y, X_predict, criterion='gini'):
	model = DecisionTreeClassifier(criterion=criterion).fit(X, y)

	training_score = model.score(X, y)
	print('Decision Tree: ', training_score)

	return model.predict(X_predict)


def random_forest(X, y, X_predict, n_estimators=10, criterion='mse', bootstrap=True):
	model = RandomForestRegressor().fit(X, y)

	training_score = model.score(X, y)
	print('Random Forests: ', training_score)

	return np.around(model.predict(X_predict))


def knn(X, y, X_predict, k=1, algorithm='auto', weights='uniform'):
	# k = int(np.sqrt(len(X)))
	# if k % 2 == 0:
	# 	k += 1
	
	model = KNeighborsClassifier(n_neighbors=k, algorithm=algorithm, weights=weights).fit(X, y)

	training_score = model.score(X, y)
	print('k Nearest Nighbors: ', training_score)

	return model.predict(X_predict)


def naive_bayes(X, y, X_predict, alpha=1, fit_prior=False):
	X = shift_data(X)
	X_predict = shift_data(X_predict)

	model = MultinomialNB(alpha=alpha, fit_prior=fit_prior).fit(X, y)

	training_score = model.score(X, y)
	print('Multinomial Naive Bayes: ', training_score)

	return model.predict(X_predict)


def adaboost(X, y, X_predict, learning_rate=10):
	base_estimator = SVC(gamma='auto', C=5, kernel='linear')
	model = AdaBoostClassifier(base_estimator=base_estimator, learning_rate=learning_rate, algorithm='SAMME').fit(X, y)

	training_score = model.score(X, y)
	print('Multinomial Naive Bayes: ', training_score)

	return model.predict(X_predict)



X_train, y_train, y_train_one_hot, X_predict = load_data()

# predictions = linear_regression(X_train, y_train, X_predict)
# predictions = logistic_regression(X_train, y_train, X_predict)
# predictions = svc(X_train, y_train, X_predict)
# predictions = decision_trees(X_train, y_train, X_predict)
# predictions = random_forest(X_train, y_train, X_predict)
# predictions = knn(X_train, y_train, X_predict)
# predictions = naive_bayes(X_train, y_train, X_predict)
predictions = adaboost(X_train, y_train, X_predict)

print(predictions)


