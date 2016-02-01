import theano
import theano.tensor as T 
import numpy as np 

class NeuralNetwork(object):

	def __init__(self, input, target,activation = T.tanh):

		self.input = input
		self.target = target
		n_sample,n_features = input.shape
		n_target = len(set(target))
		rng = np.random.RandomState(1234)

		n_in1 = n_features
		n_out1 = 30

		W1_values= np.array( 
						rng.uniform(
						low  = -np.sqrt(6./(n_in1 + n_out1)),
						high = np.sqrt(6./(n_in1 + n_out1)),
						size = (n_in1,n_out1)

					)
				
			)

		if activation == T.nnet.sigmoid:
			W1_values *= 4

		self.W1 = theano.shared(value = W1_values, name = "W1", borrow = True)
		self.b1 = theano.shared(value = np.zeros((n_out1,)), name = "b1", borrow = True)

		self.X = T.matrix("X")
		self.Y = T.lvector("Y")

		self.out1 = activation(T.dot(self.X,self.W1) + self.b1)

		W2_values = np.zeros((n_out1,n_target))

		self.W2 = theano.shared(value = W2_values, name ="W2", borrow = True)
		self.b2 = theano.shared(value = np.zeros((n_target,)), name = "b2",borrow = True)

		self.y_prob = T.nnet.softmax(T.dot(self.out1,self.W2) + self.b2)
		self.y_pred = T.argmax(self.y_prob, axis = 1)

	def fit(self,L1,L2,learning_rate =0.01):

		print "The model is being trained......."

		L1_reg = (
					abs(self.W1).sum() + abs(self.W2).sum()
			)

		L2_reg = (
					abs(self.W1 ** 2).sum() + abs(self.W2 ** 2).sum()
			)

		cost = (
				-T.mean(T.log(self.y_prob)[T.arange(self.Y.shape[0]), self.Y])
				+ L1 * L1_reg
				+ L2 * L2_reg
				)

		gW1,gb1,gW2,gb2 = T.grad(cost,[self.W1,self.b1,self.W2,self.b2])

		updates = [
					(self.W1, self.W1 - learning_rate*gW1),
					(self.b1, self.b1 - learning_rate*gb1),
					(self.W2, self.W2 - learning_rate*gW2),
					(self.b2, self.b2 - learning_rate*gb2)
			]
		train = theano.function(inputs = [self.X,self.Y], outputs = cost , updates = updates)

		learning_steps = 10000

		for i in xrange(learning_steps):
			train(self.input,self.target)

	def predict(self, test_data):

		
		prediction = theano.function(inputs = [self.X], outputs = self.y_pred)
		return prediction(test_data)


