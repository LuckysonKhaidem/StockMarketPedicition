import theano
import theano.tensor as T 
import numpy as np 

class NeuralNetwork(object):

	def __init__(self, input, target,activation = T.nnet.sigmoid):

		self.input = input
		self.target = target
		n_sample,n_features = input.shape
		n_target = len(set(target))
		rng = np.random.RandomState(1234)

		n_in1 = n_features
		n_out1 = 50

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

		W2_values= np.array( 
						rng.uniform(
						low  = -np.sqrt(6./(n_in1 + n_out1)),
						high = np.sqrt(6./(n_in1 + n_out1)),
						size = (n_out1,n_target)

					)
				
			)

		if activation == T.nnet.sigmoid:
			W2_values *= 4

		self.W2 = theano.shared(value = W2_values, name ="W2", borrow = True)
		self.b2 = theano.shared(value = np.zeros((n_target,)), name = "b2",borrow = True)

		self.y_prob = T.nnet.softmax(T.dot(self.out1,self.W2) + self.b2)
		self.y_pred = T.argmax(self.y_prob, axis = 1)

	def fit(self,L1,L2,learning_rate =0.01):

		print "The model is being trained......."
		d = {}
		L1_reg = abs(self.W1).sum() + abs(self.W2).sum()
		L2_reg = abs(self.W1 ** 2).sum() + abs(self.W2 ** 2).sum()
			
		cost = T.scalar("cost")
		cost = -T.mean(T.log(self.y_prob)[T.arange(self.Y.shape[0]), self.Y])+ L1 * L1_reg + L2 * L2_reg
				

		#cost = T.nnet.categorical_crossentropy(self.y_prob,self.Y)
		gW1,gb1,gW2,gb2 = T.grad(cost,[self.W1,self.b1,self.W2,self.b2])

		updates = [
					(self.W1, self.W1 - learning_rate*gW1),
					(self.b1, self.b1 - learning_rate*gb1),
					(self.W2, self.W2 - learning_rate*gW2),
					(self.b2, self.b2 - learning_rate*gb2)
			]

		train = theano.function(inputs = [self.X,self.Y], outputs = cost , updates = updates)
		n = self.input.shape[0]
		self.cCost = []
		for i in xrange(1000):
			for start,end in zip(range(0,n,128),range(128,n,128)):
				c = train(self.input[start:end],self.target[start:end])
				params = [self.W1.get_value(),self.b1.get_value(),self.W2.get_value(),self.b2.get_value()]
				d[float(c)] = params
				self.cCost.append(c)
				print c
		min_cost = min(d.keys())
		self.W1.set_value(d[min_cost][0])	
		self.b1.set_value(d[min_cost][1])
		self.W2.set_value(d[min_cost][2])
		self.b2.set_value(d[min_cost][3])
		
			
	def predict(self, test_data):

		
		prediction = theano.function(inputs = [self.X], outputs = self.y_pred)
		return prediction(test_data)


