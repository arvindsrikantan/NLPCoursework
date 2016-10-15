import re,nltk,csv
from nltk import word_tokenize
import numpy as np
import feature_functions as f 

r = open('data.csv','rb')
targets = []
inputs = []
tags = ['Defended','Left alone','Beaten','Edged','Caught','Runout','Stumped','Bowled','LBW','Boundary_scored_by_batsman','Runs_by_batsman','Boundary_scored_extras','Runs_by_extras','Catch_dropped','Stumping Missed','Runout Missed','Bouncer','Yorker','Overthrow','great_save','poor_fielding','Free hit']
funcs = [f.f1,f.f2,f.f3,f.f4,f.f5,f.f6,f.f7,f.f8,f.f9,f.f10,f.f11,f.f12,f.f13,f.f14,f.f15,f.f16,f.f17,f.f18,f.f19,f.f20,f.f21,f.f22]
reader = csv.reader(r)
i = 0
for row in reader:
	i += 1
	if i == 1 or i == 2:
		continue
	else:
		comment = word_tokenize(row[1])
		# print i
		inputs.append([f(comment) for f in funcs])
		targets.append(list(map(lambda x: 0 if x=='' else int(x), row[2:])))

