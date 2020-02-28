from math import exp
from numpy import array_split
from pandas import read_csv
from random import seed
from random import random
from random import randrange


# Find the min and max value for each column
# Assumes a 'dataFrame' is given
def minmax(dataset):
    minmax = []
    for colName in dataset:
        stats = [dataset[colName].min(), dataset[colName].max()]
        minmax.append(stats)
    return minmax


# Rescale values in columns to the range 0-1
# The 'class' column is not manipulated
def normalize_dataset(dataset):
    for colName in dataset:
        if colName == 'class':
            continue
        minVal = dataset[colName].min()
        maxVal = dataset[colName].max()
        dataset[colName] = (dataset[colName] - minVal) / (maxVal - minVal)
    return dataset


# Split a dataset into k folds
def cross_validation_split(dataset, nFolds):
    datasetSplit = []
    datasetCopy = dataset.values.tolist()
    foldSize = int(len(datasetCopy) / nFolds)
    for _ in range(nFolds):
        fold = []
        while len(fold) < foldSize:
            index = randrange(len(datasetCopy))
            fold.append(datasetCopy.pop(index))
        datasetSplit.append(fold)
    return datasetSplit


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / len(actual) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, nFolds, *args):
    folds = cross_validation_split(dataset, nFolds)
    scores = []
    for fold in folds:
        trainFolds = folds
        trainFolds.remove(fold)
        testSet = []
        trainSet = []
        for trainFold in trainFolds:
            for row in trainFold:
                trainSet.append(row)
        for row in fold:
            testSet.append(row)
        predicted = algorithm(trainSet, testSet, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Initialize a network
def initialize_network(nInputs, nHidden, nOutputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(nInputs + 1)]}
                    for i in range(nHidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(nHidden + 1)]}
                    for i in range(nOutputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]    # Bias weight
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
# Using sigmoid activation function
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of sigmoid function (neuron output)
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, lRate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += lRate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += lRate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, lRate, nEpoch, nOutputs):
    for epoch in range(nEpoch):
        sumError = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(nOutputs)]
            expected[int(row[-1])-1] = 1
            sumError += sum([(expected[i]-outputs[i]) **
                             2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, lRate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, lRate, sumError))


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))  # arg max function


# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, lRate, nEpoch, nHidden):
    print(train)
    print('\n\n\n\n\n\n')
    print(test)
    nInputs = len(train[0]) - 1
    nOutputs = len(set([row[-1] for row in train]))
    network = initialize_network(nInputs, nHidden, nOutputs)
    train_network(network, train, lRate, nEpoch, nOutputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return(predictions)

seed(1)
# Load and prepare dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv"
names = ['area', 'perimeter', 'compactness', 'lenght_of_kernel',
         'width_of_kernel', 'asymmetry_coefficient', 'lenght_of_kernel_grove', 'class']
dataset = normalize_dataset(read_csv(url, names=names))

# Evaluate algorithm
nFolds = 5
lRate = 0.9
nEpoch = 500
nHidden = 5

scores = evaluate_algorithm(dataset, back_propagation, nFolds, lRate, nEpoch, nHidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))