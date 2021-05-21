from math import exp
import pickle

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

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

def crop_predict(row):
    row.append(None)
    normalize_dataset([row],minmax)
    crop_index=predict(network,row)
    for key in target_index.keys():
        if target_index[key]==crop_index:
            return key
    return ""


network = pickle.load(open("network.pkl","rb"))
target_index = pickle.load(open("target_index.pkl","rb"))
minmax = pickle.load(open("minmax.pkl","rb"))

# print(target_index)
# print(crop_predict([0,19,31,25.51791333,94.38420565,6.271952832999999,178.7297725]))