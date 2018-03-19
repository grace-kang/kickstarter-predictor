import numpy
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

MAX_EPOCH = 100

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(7,1)
		
    def forward(self, x):
        x = F.relu(self.l1(x))
        return x

def percentage_correct(pred, labels, threshold = 0.5):
	correct = 0
	total = 0
	converted_pred = []
	for p in pred:
		if (p.data[0] > threshold):
			converted_pred.append(1)
		else:
			converted_pred.append(0)
            
	if (len(converted_pred) == len(labels)):
		for i in range(len(converted_pred)):
			if (converted_pred[i] == labels[i].data[0]):
				correct += 1
			total += 1
	return correct/total, correct, total
		
## Importing Data ##
dataset = pandas.read_csv('kickstarter_data_full.csv', low_memory = False)
data = dataset[[
	'disable_communication',
	'country',
	'currency',
	'staff_pick',
	'static_usd_rate',
	'category',
	'spotlight',
	'SuccessfulBool'
]].dropna().reset_index(drop = True)

## Converting Categorical Columns to Integers and Bools to 0/1 ##
data['disable_communication'] = (data['disable_communication']).astype(int)
data['staff_pick'] = (data['staff_pick']).astype(int)
data['spotlight'] = (data['spotlight']).astype(int)
data['country'] = (data['country']).astype('category').cat.codes
data['currency'] = (data['currency']).astype('category').cat.codes
data['category'] = (data['category']).astype('category').cat.codes

## Initiallizing Testing and Training Data ##
Y = data.iloc[0:int(data.size / 8), 7].as_matrix()
X = data.iloc[0:int(data.size / 8), data.columns != 'SuccessfulBool'].as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train)
y_test = torch.Tensor(y_test)

train_data  = Variable(X_train)
train_labels= Variable(y_train)
test_data   = Variable(X_test)
test_labels = Variable(y_test)

## Training the Model ##
model = Classifier()
optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.003)
loss = nn.BCEWithLogitsLoss()

accuracies = []
b_size = 100 #batch size

for epoch in range(MAX_EPOCH):
    model.train()
    for batch in range(0,train_data.size(0),b_size):
        d = train_data[batch:batch+b_size];
        l = train_labels[batch:batch+b_size]
        optimizer.zero_grad()
        pred = model(d).view(len(l))
        error = loss(pred, l)
        error.backward()
        optimizer.step()

## Testing the Model ##        
model.eval()
pred = model(test_data).view(len(test_labels))
error = loss(pred, test_labels)
acc, cor, tot = percentage_correct(pred, test_labels)
print("===================================")
print("Final Accuracy")
print(acc)
print("Correct Predictions: ", cor)
print("Total Test Labels: ", tot)
print("===================================")