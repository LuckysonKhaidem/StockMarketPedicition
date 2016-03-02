import yahoo_finance
import urllib2
from datetime import datetime

class DataFetcher:

	def __init__(self):
		pass

	def getHistoricalData(self,StockSymbol):

		currentDay = datetime.now().day 
		currentMonth = datetime.now().month
		currentYear = datetime.now().year

		url = "http://real-chart.finance.yahoo.com/table.csv?s="+StockSymbol+"&d="+str(currentMonth-1)+"&e="+str(currentDay)+"&f="+str(currentYear)+"&g=d&a=0&b=1&c=1900&ignore=.csv"
		try:
			url_handle = urllib2.urlopen(url)
		except:
			return -1

		csv_data = url_handle.read()

		outputfile = StockSymbol+".csv"

		with open(outputfile,"w") as file:
			file.write(csv_data)

		return 1



		