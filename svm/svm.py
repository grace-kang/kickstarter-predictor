import numpy
import pandas
import torch
from torch.autograd import Variable
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

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


## Importing Data ##
dataset = pandas.read_csv('../data/kickstarter_data_full.csv', low_memory = False)
data = dataset[[
	'disable_communication',
	'country',
	'currency',
	'staff_pick',
	'static_usd_rate',
	'category',
	# 'spotlight',
	'SuccessfulBool'
]].dropna().reset_index(drop = True)

## Converting Categorical Columns to Integers and Bools to 0/1 ##
data['disable_communication'] = (data['disable_communication']).astype(int)
data['staff_pick'] = (data['staff_pick']).astype(int)
# data['spotlight'] = (data['spotlight']).astype(int)
data['country'] = (data['country']).astype('category').cat.codes
data['currency'] = (data['currency']).astype('category').cat.codes
data['category'] = (data['category']).astype('category').cat.codes

## Initiallizing Testing and Training Data ##
Y = data.iloc[0:int(data.size / 7), 6].as_matrix()
X = data.iloc[0:int(data.size / 7), data.columns != 'SuccessfulBool'].as_matrix()

X_normalized = normalize(X, norm='l2')

X_train, X_test, y_train, y_test = train_test_split(X_normalized, Y, test_size = 0.2, random_state = 42)

X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train)
y_test = torch.Tensor(y_test)

train_data  = Variable(X_train)
train_labels= Variable(y_train)
test_data   = Variable(X_test)
test_labels = Variable(y_test)

#initialize SVM
clf = svm.SVC()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
acc, cor, tot = percentage_correct(pred, y_test)
print ('Final Accuracy: {}'.format(acc))

