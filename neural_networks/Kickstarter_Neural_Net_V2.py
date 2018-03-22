import numpy
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

MAX_EPOCH = 2000
PRED_THRESHOLD = 0.5

torch.manual_seed(474)
torch.backends.cudnn.enabled = False

def percentage_correct(pred, labels):
	correct = 0
	total = 0
	converted_pred = []
	for p in pred:
		if (p.data[0] > PRED_THRESHOLD):
			converted_pred.append(1)
		else:
			converted_pred.append(0)
            
	if (len(converted_pred) == len(labels)):
		for i in range(len(converted_pred)):
			if (converted_pred[i] == labels[i].data[0]):
				correct += 1
			total += 1
	return correct/total, correct, total

######
#   Architecture of the neural network.
######
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(60,30)
        self.l2 = nn.Linear(30,15)
        self.l3 = nn.Linear(15,5)
        self.l4 = nn.Linear(5,1)
		
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.sigmoid(self.l4(x))
        return x

######
#   Helper class for interacting with the neural network from other programs.
######
class KickstarterNeuralNet:

    ######
    #   Takes a given Pandas dataframe, trains the neural network on it and then
    #   adds the predictions for all of the data to the dataframe and returns it.
    #
    #   train_data & test_data: Pandas dataframes containing at least the following column:
    #       -   disable_communication
	#       -   country
	#       -   currency
	#       -   staff_pick
	#       -   static_usd_rate
	#       -   category
	#       -   SuccessfulBool
    #
    ######
    def __init__(self, train_data, test_data):
        self.original_train_data = train_data
        self.original_test_data = test_data
        self.model = None
        self.error_data = []
        self.accuracy_data = []
        
        x_train, y_train = self.extract_features_and_labels(self.original_train_data)
        x_test, y_test = self.extract_features_and_labels(self.original_test_data)
        
        # Converting to variables.
        x_train = torch.Tensor(x_train)
        x_test = torch.Tensor(x_test)
        y_train = torch.Tensor(y_train)
        y_test = torch.Tensor(y_test)
        self.train_data  = Variable(x_train)
        self.test_data   = Variable(x_test)
        self.train_labels= Variable(y_train)
        self.test_labels = Variable(y_test)
        
    ######
    #   Extracts the features and labels from the given dataframe and 
    #   performs one-hot encoding for all categorical data.
    #
    #   Returns:
    #       -   Data
    #       -   Labels
    ######
    def extract_features_and_labels (self, dataframe):
        # Getting only the features we are interested in.
        dataframe_subset = dataframe[[
            'disable_communication',
            'country',
            'currency',
            'staff_pick',
            'static_usd_rate',
            'category',
            'SuccessfulBool'
        ]].reset_index(drop = True)
        
        # Converting booleans to 0/1.
        dataframe_subset['disable_communication'] = (dataframe_subset['disable_communication']).astype(int)
        dataframe_subset['staff_pick'] = (dataframe_subset['staff_pick']).astype(int)
        
        # Performing one hot encoding for categorical data.
        country_one_hot = pandas.get_dummies(dataframe_subset['country'])
        currency_one_hot = pandas.get_dummies(dataframe_subset['currency'])
        category_one_hot = pandas.get_dummies(dataframe_subset['category'])
        dataframe_subset = dataframe_subset.drop('country', axis = 1)
        dataframe_subset = dataframe_subset.drop('currency', axis = 1)
        dataframe_subset = dataframe_subset.drop('category', axis = 1)
        dataframe_subset = dataframe_subset.join(country_one_hot)
        dataframe_subset = dataframe_subset.join(currency_one_hot)
        dataframe_subset = dataframe_subset.join(category_one_hot)
        
        # Splitting the data from the labels.
        X = dataframe_subset.drop('SuccessfulBool', axis = 1).as_matrix()
        Y = dataframe_subset['SuccessfulBool'].as_matrix()
        
        return X, Y
    
    ######
    #   Trains the neural network on the provided data and then tests it.
    #
    #   Returns:
    #       -   Final accuracy from testing on the test data.
    #       -   Final error from the testing on the test data.
    ######
    def train_and_test(self):
    
        # Initializing the model.
        self.model = Classifier()
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr = 0.003)
        loss = nn.L1Loss()

        # Training the model.
        for epoch in range(MAX_EPOCH):
            self.model.train()
            optimizer.zero_grad()
            pred = self.model(self.train_data).view(len(self.train_labels))
            error = loss(pred, self.train_labels)
            error.backward()
            optimizer.step()
            
            self.model.eval()
            pred = self.model(self.test_data).view(len(self.test_labels))
            self.error_data.append(loss(pred, self.test_labels).data[0])
            #accuracy, correct, total  = percentage_correct(pred, self.test_labels)
            #self.accuracy_data.append(accuracy)
            
        # Testing the model.
        self.model.eval()
        pred = self.model(self.test_data).view(len(self.test_labels))
        final_error = loss(pred, self.test_labels)
        final_accuracy, correct, total  = percentage_correct(pred, self.test_labels)
        
        return final_accuracy, final_error.data[0]
    
    ######
    #   Plots the error per epoch from when the model was training.
    ######
    def plot_error(self):
        plt.plot(self.error_data)
        plt.title("Error per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.show()
        
    ######
    #   Plots the accuracy per epoch from when the model was training.
    ######
    def plot_accuracy(self):
        plt.plot(self.accuracy_data)
        plt.title("% Accuracy per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("% Accuracy")
        plt.show()
        
    ######
    #   Uses the trained model to predict the success of a campaign for the
    #   entire dataset that was given. It then adds these predictions to the
    #   original dataset as a new column called 'nn_prediction' and returns
    #   the dataset. If you set 'save_to_csv' to True and pass in a file name
    #   it will also save the new dataframe to a csv.
    #
    #   save_to_csv: Set to True if you want it to save the new dataset to a csv.
    #
    #   file_name: A string of the name of the csv file.
    #       - Example: 'test_file.csv'
    ######
    def get_dataframe(self, save_to_csv = False, file_name = None):
        
        # Making predictions.
        self.model.eval()
        pred = self.model(self.test_data).view(len(self.test_labels))
        
        # Adding the predictions to the dataframe
        self.original_test_data['nn_prediction'] = pred.data.numpy()
        
        if save_to_csv and file_name:
            self.original_test_data.to_csv(file_name, encoding='utf-8', index=False)
        
        return self.original_test_data

######
#   If this file is called directly the following will occur.
######
if __name__ == "__main__":
    train_dataset = pandas.read_csv('../data/train_data.csv', low_memory = False)
    test_dataset = pandas.read_csv('../data/test_data.csv', low_memory = False)
    
    ks_nn = KickstarterNeuralNet(train_dataset, test_dataset);
    final_accuracy, final_error = ks_nn.train_and_test();
    
    print("== Final Accuracy ==")
    print(numpy.around(final_accuracy, 3))
    print("== Final Error ==")
    print(numpy.around(final_error, 3))
    
    ks_nn.plot_error()
    #ks_nn.plot_accuracy()
    
    df = ks_nn.get_dataframe(True, "test_data_with_nn_predictions.csv")    