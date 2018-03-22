import numpy
import pandas
import torch
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

def extract_data(dataset):
	data = dataset[[
		'disable_communication',
		'country',
		'currency',
		'staff_pick',
		'static_usd_rate',
		'category',
		'SuccessfulBool'
	]].reset_index(drop = True)

	# Converting booleans to 0/1.
	data['disable_communication'] = (data['disable_communication']).astype(int)
	data['staff_pick'] = (data['staff_pick']).astype(int)

	# Performing one hot encoding for categorical data.
	country_one_hot = pandas.get_dummies(data['country'])
	currency_one_hot = pandas.get_dummies(data['currency'])
	category_one_hot = pandas.get_dummies(data['category'])
	data = data.drop('country', axis = 1)
	data = data.drop('currency', axis = 1)
	data = data.drop('category', axis = 1)
	data = data.join(country_one_hot)
	data = data.join(currency_one_hot)
	data = data.join(category_one_hot)

	X = data.drop('SuccessfulBool', axis = 1).as_matrix()
	Y = data['SuccessfulBool'].as_matrix()
	return X, Y


def percentage_correct(pred, labels, threshold = 0.5):
	correct = 0
	total = 0
	converted_pred = []
	for p in pred:
		if (p > threshold):
			converted_pred.append(1)
		else:
			converted_pred.append(0)
            
	if (len(converted_pred) == len(labels)):
		for i in range(len(converted_pred)):
			if (converted_pred[i] == labels[i]):
				correct += 1
			total += 1
	return correct/total, correct, total



##import and extract train and test data
train_data = pandas.read_csv('../data/train_data.csv', low_memory = False)
test_data = pandas.read_csv('../data/test_data.csv', low_memory = False)
X_train, y_train = extract_data(train_data)
X_test, y_test = extract_data(test_data)

#initialize SVM and train model
clf = svm.SVC(C=100, kernel='rbf')
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
pred = clf.predict(X_test)
acc, cor, tot = percentage_correct(pred, y_test)
print ('Final Accuracy: {}'.format(acc))

#make predictions for whole dataset for new column
whole_dataset = pandas.read_csv('../data/kickstarter_data_full.csv', low_memory = False)
X, Y = extract_data(whole_dataset)
#pred_col = clf.predict(X)
#original_data['svm_prediction'] = pred_col
#original_data.to_csv('data_with_svm_predictions.csv')

