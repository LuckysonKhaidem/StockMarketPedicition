class DecisionNode:

	def __init__(self,
				col = -1,
				value = None,
				isLeaf = False,
				results = None,
				tb = None,
				fb = None):

		self.isLeaf = isLeaf
		self.col = col
		self.value = value
		self.results = results
		self.true_branch = tb
		self.false_branch = fb
