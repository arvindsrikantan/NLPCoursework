import re,nltk,csv
from nltk import word_tokenize
from my_maxent_test import MyMaxEnt
import numpy as np
import feature_functions as f 

if __name__ == '__main__':
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
			# print i
			comment = tuple(word_tokenize(row[1]))
			inputs.append(tuple([comment,row[2:]]))

	# create multiple max ent classifers 
	c1 = MyMaxEnt(inputs,funcs,[1 if tags[i] in ['Defended','Left alone','Beaten'] else 0 for i in range(len(tags))])
	c1.train()
	c1.save('c1_model.pickle')
	print c1.classify(inputs[0][0])
	print inputs[0][1]
	# c2 = 