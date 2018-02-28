import numpy
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.autograd import Variable


MAX_EPOCH = 10

class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()
		self.l1 = nn.Linear(7,1)
		
	def forward(self, x):
		x = F.sigmoid(self.l1(x))
		return x

def percentage_correct(pred, labels, threshold = 0.5):
	correct = 0
	total = 0
	converted_pred = []
	for p in pred:
		if (p.data[0] > 0.5):
			converted_pred.append(1)
		else:
			converted_pred.append(0)
            
	if (len(converted_pred) == len(labels)):
		for i in range(len(converted_pred)):
			if (converted_pred[i] == labels[i].data[0]):
				correct += 1
			total += 1
	return correct/total
		
## Getting the Data Ready ##
dataset = pandas.read_csv('kickstarter_data_full.csv')
data = dataset[[
	'disable_communication',
	'country',
	'currency',
	'staff_pick',
	'static_usd_rate',
	'category',
	'spotlight',
	'successfulbool'
]].dropna().reset_index(drop = True)

Y = data.iloc[0:int(data.size / 8), 7].as_matrix()
X = data.iloc[0:int(data.size / 8), data.columns != 'successfulbool'].as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

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
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
loss = nn.BCELoss()

b_size = 100 #batch size
for epoch in range(MAX_EPOCH):
    model.train()
    for batch in range(0,train_data.size(0),b_size):
        pred = model(train_data[batch:batch+b_size])
        error = loss(pred, train_labels[batch:batch+b_size])
        optimizer.zero_grad()
        error.backward()
        #print("\t[training batch {:3d}/{:3d}]".format(int(batch/b_size), int(train_data.size(0)/b_size)),error.data[0])
        optimizer.step()
    print("#"*64+"\nAccuracy")
    model.eval()
    pred = model(test_data)
    error = loss(pred, test_labels)
    #print("EPOCH: ",epoch, error.data[0])
    print(percentage_correct(pred, test_labels))
    print("#"*64)