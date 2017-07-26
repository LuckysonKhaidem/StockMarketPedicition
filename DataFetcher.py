import urllib
from datetime import datetime
import os
# import mechanize
import requests

class DataFetcher:

	def __init__(self):
		pass

	def getHistoricalData(self,stock_symbol):
		
		currentDay = datetime.now().day 
		currentMonth = datetime.now().month
		currentYear = datetime.now().year
		url = "http://real-chart.finance.yahoo.com/table.csv?s="+stock_symbol+"&d="+str(currentMonth-1)+"&e="+str(currentDay)+"&f="+str(currentYear)+"&g=d&a=0&b=1&c=1900&ignore=.csv"		
	
		dir_name = os.path.dirname(os.path.abspath(__file__))
		filename = stock_symbol+".csv"
		outputfile = os.path.join(dir_name,"Dataset",filename)
		
		if os.path.isfile(outputfile):
			timestamp = os.path.getmtime(outputfile)
			lastmodifiedtime = datetime.fromtimestamp(timestamp)
			if (currentDay == lastmodifiedtime.day and 
				currentMonth == lastmodifiedtime.month and 
				currentYear == lastmodifiedtime.year):
				return 1 
		try:
			content = urllib.urlopen(url).read()
			with open(outputfile,"w") as f:
				f.write(content)
			return 1
		except:
			return -1




