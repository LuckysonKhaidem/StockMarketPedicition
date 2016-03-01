from threading import Thread 

class SplittingThread(Thread):

	def __init__(self, func, argument):
		Thread.__init__(self)
		self.return_value = None
		self.func = func
		self.argument = argument

	def run(self):
		self.return_value = self.func(self.argument)
	def join(self):
		Thread.join(self)
		return self.return_value

