#handwritten digit recognition
import numpy as np
import matplotlib.pyplot as plt
import scipy.special

class NeuralNetwork:

	def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
		#nodes
		self.iNodes = inputNodes
		self.hNodes = hiddenNodes
		self.oNodes = outputNodes

		#weights between input & hidden and hidden & output
		self.wih = np.random.rand(self.hNodes, self.iNodes) - .5
		self.who = np.random.rand(self.oNodes, self.hNodes) - .5

		#learning rate
		self.lRate = learningRate

		#activation (sigmoid)
		self.activation = lambda x: scipy.special.expit(x)
		pass

	def train(self, input_list, target_list):
		#converting input and target lists to transposed(.T) 2D arrays
		inputs = np.array(input_list, ndmin=2).T
		targets = np.array(target_list, ndmin=2).T

		#activated hidden layer:
		hInputs = np.dot(self.wih, inputs)
		hLayer = self.activation(hInputs)

		#activated final layer:
		fInputs = np.dot(self.who, hLayer)
		fLayer = self.activation(fInputs)

		#errors for refining weights between hidden and output
		outputErrors = targets - fLayer

		#errors for refining weights between input and hidden
		hiddenErrors = np.dot(self.who.T, outputErrors)

		#update weights between hidden and output
		self.who += self.lRate * np.dot(outputErrors * fLayer * (1 - fLayer), hLayer.T)

		#update weights between input and hidden
		self.wih += self.lRate * np.dot(hiddenErrors * hLayer * (1 - hLayer), inputs.T)

	def query(self, input_list):
		#convert input_list into a transposed 2D array
		inputs = np.array(input_list, ndmin=2).T

		#activated hidden layer:
		hLayer = self.activation(np.dot(self.wih, inputs))

		#activated final layer:
		fLayer = self.activation(np.dot(self.who, hLayer))

		return fLayer


#object of the NN class
inp = 784
hid = 150
out = 10
lr = 0.4
NN = NeuralNetwork(inp, hid, out, lr)

#loading the MNIST file
train_data_file = open("/mnist_data_set/mnist_train_100.csv", 'r')
train_data_list = train_data_file.readlines()
train_data_file.close()

epochs = 70

#training
for i in range(epochs):
	for record in train_data_list:
		all_values = record.split(',')
		inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
		#target
		targets = np.zeros(out) + 0.01	
		targets[int(all_values[0])] = 0.99
		NN.train(inputs, targets)
	print("{}%".format(int(i*(100/epochs))))


#testing
test_data_file = open("/mnist_data_set/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

IMAGE_NUMBER = 9

x = test_data_list[IMAGE_NUMBER].split(',')
y = ((np.asfarray(x[1:]) / 255.0 * 0.99) + 0.01)

image_array = plt.imshow(y.reshape(28, 28), cmap='Greys')
plt.show()

prediction_list = [i[0] for i in NN.query(y)]
prediction = prediction_list.index(max(prediction_list))
print("The number is {}".format(prediction))
print("I am {}% sure.".format(prediction_list[prediction]*100))