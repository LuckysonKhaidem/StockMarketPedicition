import os

class ArtificialTrader:

	def __init__(self,current_data,Trading_Day,selected_stock):
		
		self.price = current_data[3]
		self.Trading_Day =Trading_Day
		self.selected_stock = selected_stock
		self.info_file = self.selected_stock+"trade_info"

		if not os.path.exists(self.info_file):
			f = open(self.info_file,"w")
			f.write("0,0")
			f.close()



	def trade(self,prediction):

		def buy():
			print "The price will rise after",self.Trading_Day,"Days"
			print "Suggested Action: Buy"

			f = open(self.info_file,"r")
			current_gain,number_of_stocks = map(float,f.read().split(","))
			current_gain -= self.price
			number_of_stocks += 1
			f.close()
			print "Current Profit",current_gain
			print "Number of stocks bought",number_of_stocks
			f = open(self.info_file,"w")
			f.write(str(current_gain)+","+str(number_of_stocks))
			f.close()

		def sell():
			print "The price will fall after",self.Trading_Day,"Days"
			print "Suggestd Action: Sell"

			f = open(self.info_file,"r")
			current_gain,number_of_stocks = map(float,f.read().split(","))
			current_gain += current_data[3]*number_of_stocks
			number_of_stocks = 0 
			f.close()
			print "Current Profit",current_gain
			print "Number of stocks bought",number_of_stocks
			f = open(self.info_file,"w")
			f.write(str(current_gain)+","+str(number_of_stocks))
			f.close()

		if isinstance(prediction,list):
			prediction = prediction[0]

		if prediction == 1:
			buy()
		else:
			sell() 