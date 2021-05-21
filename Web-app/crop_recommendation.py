# Backprop on the Seeds Dataset
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
import pickle

# Initialize a network
# each node is implemented using dictionary
# each layer is a list of nodes, and network is a list of layers
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
# basically weighted sum of inputs to the neuron
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
# sigmoid function
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = [] # store input for next layer/ output of this layer

        # traverse through each node and calculate its output
		for neuron in layer: # each neuron is a dictionary
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation) # storing output in the dictionary/neuron
			new_inputs.append(neuron['output'])

		inputs = new_inputs

	return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
            # for hidden layers
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
            # for output layer
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1] # excluding the target value
		if i != 0:
            # for layers other than first hidden layer take input from previous layer
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
                # update each weight for the neuron
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            # updating the bias weight
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
# epoch refers to one full iteration of the dataset
# l_rate is the learning rate
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row) # get output using current weights
            
            # since the target/class value is known, probability for that value is 1 and rest is 0
			expected = [0 for i in range(n_outputs)] # initialize to zero
			expected[row[-1]] = 1 # set the class value to 1
            
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
        
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	pickle.dump(network,open("model.pkl","wb"))
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)


# Test Backprop on Seeds dataset
seed(1)
# load and prepare data
filename = 'Crop_recommendation.csv'
dataset = load_csv(filename)
dataset.pop(0)

# target_set = set(row[-1] for row in dataset)
# target_index = {}
# i = 0
# for target in target_set :
#     target_index[target] = i
#     i+=1

# pickle.dump(target_index,open("target_index.pkl","wb"))

target_index = pickle.load(open("target_index.pkl","rb"))

# replace class labels (strings) with integer equivalent (index)
for row in dataset:
    for i in range(len(row)-1):
        row[i] = float(row[i])
    row[-1] = int(target_index[row[-1]])

# normalize input variables
minmax = dataset_minmax(dataset)
# normalize_dataset(dataset, minmax)

pickle.dump(minmax,open("minmax.pkl","wb"))

# evaluate algorithm
# n_folds = 5
# l_rate = 0.3
# n_epoch = 500
# n_hidden = 15
# scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
# print('Scores: %s' % scores)
# print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

# class Model:
# 	def __init__(self,network,target_index,minmax):
# 		self.network = network
# 		self.target_index = target_index
# 		self.minmax = minmax
	
# 	# Make a prediction with a network
# 	def predict(self,row):
# 		outputs = self.forward_propagate(row)
# 		return outputs.index(max(outputs))

# 	# Calculate neuron activation for an input
# 	# basically weighted sum of inputs to the neuron
# 	def activate(self,weights, inputs):
# 		activation = weights[-1]
# 		for i in range(len(weights)-1):
# 			activation += weights[i] * inputs[i]
# 		return activation

# 	# Transfer neuron activation
# 	# sigmoid function
# 	def transfer(self,activation):
# 		return 1.0 / (1.0 + exp(-activation))

# 	# Forward propagate input to a network output
# 	def forward_propagate(self,row):
# 		inputs = row
# 		for layer in self.network:
# 			new_inputs = [] # store input for next layer/ output of this layer

# 			# traverse through each node and calculate its output
# 			for neuron in layer: # each neuron is a dictionary
# 				activation = self.activate(neuron['weights'], inputs)
# 				neuron['output'] = self.transfer(activation) # storing output in the dictionary/neuron
# 				new_inputs.append(neuron['output'])

# 			inputs = new_inputs

# 		return inputs
	
# 	# Rescale dataset columns to the range 0-1
# 	def normalize_dataset(self,dataset):
# 		for row in dataset:
# 			for i in range(len(row)-1):
# 				row[i] = (row[i] - self.minmax[i][0]) / (self.minmax[i][1] - self.minmax[i][0])

# 	def crop_predict(self,row):
# 		self.normalize_dataset([row])
# 		crop_index=self.predict(row)
# 		for key in self.target_index.keys():
# 			if self.target_index[key]==crop_index-1:
# 				return key
# 		return ""

# temp_network = pickle.load(open("model.pkl","rb"))
# model = Model(temp_network,target_index,minmax)
# pickle.dump(model,open("predict.pkl","wb"))
# print(str(model.crop_predict([75,56,18,19.39851734,62.35750641,5.696205468,60.95197486,0])))

# '''
# pickle.dump(crop_predict,open("predict.pkl","wb"))
# print(crop_predict(network,[24,67,22,20.120043,22.89845607,5.618844277000001,104.6252153]))
# '''