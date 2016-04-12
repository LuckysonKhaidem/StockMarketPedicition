import yahoo_finance
import urllib
from datetime import datetime
import mechanize
class DataFetcher:

	def __init__(self):
		pass

	def getHistoricalData(self,StockSymbol):

		currentDay = datetime.now().day 
		currentMonth = datetime.now().month
		currentYear = datetime.now().year

		outputfile = StockSymbol+".csv"
		url = "http://real-chart.finance.yahoo.com/table.csv?s="+StockSymbol+"&d="+str(currentMonth-1)+"&e="+str(currentDay)+"&f="+str(currentYear)+"&g=d&a=0&b=1&c=1900&ignore=.csv"
		try:
			urllib.urlretrieve(url,outputfile)
		except:
			return -1

	#	csv_data = url_handle.read()
	#
		#outputfile = StockSymbol+".csv"
#
#		with open(outputfile,"w") as file:
#			file.write(csv_data)
		return 1

	def getCurrentData(self,StockSymbol,DataOption):

		stock = yahoo_finance.Share(StockSymbol)

		stock.refresh()

		if DataOption == "Open":

			return stock.get_open()

		if DataOption == "Price":

			return stock.get_price()

		if DataOption == "Volume":

			return stock.get_volume()

		if DataOption == "High":

			return stock.get_days_high()

		if DataOption == "Low":

			return stock.get_days_low()

		if DataOption == "ohclv":

			return [float(stock.get_open()),
					float(stock.get_days_high()),
					float(stock.get_days_low()),
					float(stock.get_price()), 
					float(stock.get_volume())]
		else:
			raise Error("Invalid Option")







		
