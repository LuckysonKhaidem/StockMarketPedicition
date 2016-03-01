from threading import Thread 

class TreeThread(Thread):

	def __init__(self, func, X,y):
		Thread.__init__(self)
		self.return_value = None
		self.func = func
		self.X = X
		self.y = y

	def run(self):
		self.return_value = self.func(self.X,self.y)
		
	def join(self):
		Thread.join(self)
		return self.return_value

