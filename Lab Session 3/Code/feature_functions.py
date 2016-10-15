import nltk,re
def f1(h,t):
	'''
		Check if first is capital
	'''
	if (h[2][h[3]].istitle()) and (t == "PERSON" or t == "ORGANIZATION" or t == "GPE"):
		return 1
	else:
		return 0
		
def f2(h,t):
	'''
		use dictionary for organization enumeration
	'''
	dic = ["alcatel","amazon","apple","asus","blackberry","byond","coolpad","gionee","google","hp","honor","htc","huawei","iball","infocus","intex","jolla","karbonn","lava","lenovo","lg","micromax","microsoft","motorola","nokia","oneplus","samsung","oppo","panasonic","penta","phicomm","philips","sony","spice","vivo","wickedleak","xiaomi","xolo","yu","zte","kingston","pebble","strontium","sunstrike","sandisk"]
	# print h[2][h[3]]
	if (h[2][h[3]].lower() in dic) and (t == "ORGANIZATION"):
		return 1
	else:
		return 0

def f3(h,t):
	'''
		GPE if previous is "in" or "at"
	'''
	if (h[2][h[3]-1] in ["in","at"]) and t == "GPE":
		return 1
	else:
		return 0

def f4(h,t):
	'''
		Money regex
	'''
	dic = []
	# print re.match(r'\d+',h[2][h[3]]), h[2][h[3]]
	if(re.match(r'\d+',h[2][h[3]]) and(h[2][h[3]-1] in ["$","Rs"] or h[2][h[3]-2] in ["$","Rs"])) and t == "MONEY":
		return 1
	else:
		return 0
		
def f5(h,t):
	'''
		Organization followed by inc or corp
	'''
	#print h[2][h[3]+1]
	if(h[3]+1<len(h[2]) and h[2][h[3]+1] in ["inc.","corp","inc","corp."]) and t == "ORGANIZATION":
		return 1
	else:
		return 0	
		
def f6(h,t):
	if(re.search('^(([0-1]?[0-9])|([2][0-3])):([0-5]?[0-9])(:([0-5]?[0-9]))?$',h[2][h[3]]) or re.search('^((0?[13578]|10|12)(-|\/)(([1-9])|(0[1-9])|([12])([0-9]?)|(3[01]?))(-|\/)((19)([2-9])(\d{1})|(20)([01])(\d{1})|([8901])(\d{1}))|(0?[2469]|11)(-|\/)(([1-9])|(0[1-9])|([12])([0-9]?)|(3[0]?))(-|\/)((19)([2-9])(\d{1})|(20)([01])(\d{1})|([8901])(\d{1})))$',h[2][h[3]])) and t == "DATE":
		return 1
	else:
		return 0

def f7(h,t):
	'''
		Organization followed by product model
	'''
	for i in range(1,4):
		if h[3]+i<len(h[2]) and (re.match(r'\d*\[a-zA-Z]+\d+',h[2][h[3]+i])) and t == "ORGANIZATION":
			return 1
	else:
		return 0

def f8(h,t):
	'''
		City, State/Country eg: Bangalore, India
	'''
	if(h[0]=="GPE" and h[2][h[3]-1]==",") and t == "GPE":
		return 1
	else:
		return 0
		
def f9(h,t):
	'''
		Person if next word is said,rebutted .....
	'''
	if(h[3]+1<len(h[2]) and h[2][h[3]+1] in ["said","rebutted","told"]) and t == "PERSON":
		return 1
	else:
		return 0	
		
def f10(h,t):
	'''
		if in a window of 5 there is a stem with cost or price
	'''
	porter_stemmer = nltk.PorterStemmer()
	for i in range(h[3]-5,h[3]):
		#print h[2]
		if (porter_stemmer.stem(h[2][i]) in ["price","pay","paid","cost","payment","buy","bought"]) and (t == "MONEY"):
			return 1
	else:
		return 0		

def f11(h,t):
	'''
		for OTHER
	'''
	if (not h[2][h[3]].istitle() or h[2][h[3]-1]==".") and (t == "OTHER"):
		return 1
	else:
		return 0

if __name__=="__main__":
	print f10(("LOCATION","t1",nltk.word_tokenize("Hello there Ashwin is in Bangalore, India Microsoft bought X50 for Rs. 100."),14),"MONEY")