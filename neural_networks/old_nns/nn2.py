import numpy
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from sklearn.utils import shuffle

MAX_EPOCH = 1000

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(7,77)
        self.l2 = nn.Linear(77,77)
        self.l3 = nn.Linear(77,77)
        self.l4 = nn.Linear(77,77)
        self.l5 = nn.Linear(77,77)
        self.l6 = nn.Linear(77,77)
        self.l7 = nn.Linear(77,77)
        self.l8 = nn.Linear(77,2)
        self.l9 = nn.Linear(2,1)
		
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        x = F.relu(self.l7(x))
        x = F.relu(self.l8(x))
        x = F.relu(self.l9(x))
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
	return correct/total
		
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

#shuffle X and y
X_train, y_train = shuffle(X_train, y_train, random_state=0)

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.03, momentum=0.9)
loss = nn.BCEWithLogitsLoss()

accuracies = []
b_size = 100 #batch size

for epoch in range(MAX_EPOCH):
    model.train()
    for batch in range(0,train_data.size(0),b_size):
    	d = train_data[batch:batch+b_size]
    	l = train_labels[batch:batch+b_size]
    	optimizer.zero_grad()
    	pred = model(d).view(len(l))
    	error = loss(pred, l)
    	error.backward()
    	optimizer.step()
    print ('epoch {} -- percentage correct: {}, error: {}'.format(epoch, percentage_correct(pred,l), error.data[0]))


## Testing the Model ##        
model.eval()
pred = model(test_data).view(len(test_labels))
error = loss(pred, test_labels)
print("===================================")
print("Final Accuracy")
print(percentage_correct(pred, test_labels))
print("===================================")