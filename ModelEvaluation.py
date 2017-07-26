import os
import numpy as np
from sklearn.metrics import roc_curve,auc, confusion_matrix
from matplotlib import pyplot as plt
from matplotlib import animation
from sklearn.ensemble import RandomForestClassifier

class Evaluator:

	def __init__(self,xtest,ytest,y_pred,LearningModel):
		self.xtest = xtest
		self.ytest = ytest
		self.y_pred = y_pred
		self.model = LearningModel

	def confusionMatrix(self):

		self.confusion_matrix = confusion_matrix(self.ytest, self.y_pred)

	def getPerformanceMetrics(self):

		self.confusionMatrix()

		accuracy = (
					float((self.confusion_matrix[0][0]+self.confusion_matrix[1][1]))/
					(self.confusion_matrix[0][0]+self.confusion_matrix[0][1]+self.confusion_matrix[1][0]+self.confusion_matrix[1][1])
			)
		precision = (
					float((self.confusion_matrix[1][1]))/
					(self.confusion_matrix[1][1] + self.confusion_matrix[0][1])
			)
		recall = (
				float((self.confusion_matrix[1][1]))/
				(self.confusion_matrix[1][1]+self.confusion_matrix[1][0])
			)
		specificity = (
				float((self.confusion_matrix[0][0]))/
				(self.confusion_matrix[0][0] + self.confusion_matrix[0][1])
			)

		return accuracy, recall, precision, specificity

	def drawROC(self):
	
		base_dir = os.path.dirname
		abspath = os.path.abspath
		dir_name =  base_dir(base_dir(base_dir(abspath(__file__))))

		y_prob = self.model.predict_proba(self.xtest)
	
		true_probability_estimate = y_prob[:,1]
	
		fpr,tpr,threshold = roc_curve(self.ytest,true_probability_estimate)
		area = auc(fpr,tpr)
		plt.figure()
		plt.plot(fpr,tpr,linewidth = 2.0,label = "ROC curve (Area= %0.2f)" % area)
		plt.plot([0,1],[0,1],"r--")
		plt.xlabel("False Postive Rate")
		plt.ylabel("True Positive Rate")
		plt.legend(loc = "lower right")
		plt.show(block = False)
		
		#plt.savefig(savepath)
		#plt.close()

	def oob_vs_n_trees(self,max_trees,Xtrain, ytrain):

		# First set up the figure, the axis, and the plot element we want to animate
		print ""
		print "Number of Trees\t\tOOB Error Rate"
		fig = plt.figure()
		ax = plt.axes(xlim=(0, max_trees), ylim=(0,1))
		line, = ax.plot([], [], lw=2)

		# initialization function: plot the background of each frame
		def init():
		    line.set_data([], [])
		    return line,

		number_of_trees = range(2,max_trees + 1)
		oob_errors = []
		# animation function.  This is called sequentially
		def animate(i):
			
			model = RandomForestClassifier(warm_start = True, oob_score = True, n_estimators = i)
			model.fit(Xtrain,ytrain)
			oob_error = 1 - model.oob_score_
			oob_errors.append(oob_error)
			print "{}\t\t\t{}".format(i,oob_error)

			line.set_data(number_of_trees[:len(oob_errors)], oob_errors)
			return line,

		# call the animator.  blit=True means only re-draw the parts that have changed.
		anim = animation.FuncAnimation(fig, animate, init_func=init, frames=number_of_trees, interval=100, blit=True, repeat = False)
		plt.xlabel("Number of trees")
		plt.ylabel("OOB error")
		plt.show()
		



	
		# for i in xrange(2,max_trees + 1):
		# 	model = RandomForestClassifier(warm_start = True, 
		# 		oob_score = True, 
		# 		n_estimators = i)
		# 	model.fit(Xtrain,ytrain)
		# 	oob_error = 1 - model.oob_score_
		# 	oob_errors.append(oob_error)
		# 	print i,oob_error
	

	def plotClassificationResult(self):
		self.confusionMatrix()
		x = [i + 3.0 for i in xrange(4)]
		xlabel = ["TP","FN","FP","TN"]
		plt.figure()
		plt.grid(True)
		plt.bar(x,self.confusion_matrix.reshape(-1), color= np.random.random((4,3)))
		plt.xticks([i + 3.0 for i in xrange(4)],xlabel)
		plt.show(block = False)




