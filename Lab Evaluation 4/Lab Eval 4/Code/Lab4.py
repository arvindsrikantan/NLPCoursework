import os,re
from sentence2vec.word2vec import Word2Vec, Sent2Vec, LineSentence
from nltk.tokenize import *
import pickle
import numpy as np
import csv

def create_dataset():
	f = open("cleanedTweets.txt","r").read().split("\n")

	f1 = open("50.txt","w")
	for line in f[:50]:
		f1.write(line+"\n")
	f1.close()

	f1 = open("100.txt","w")
	for line in f[50:150]:
		f1.write(line+"\n")
	f1.close()

	f1 = open("300.txt","w")
	for line in f[150:450]:
		f1.write(line+"\n")
	f1.close()

	f1 = open("500.txt","w")
	for line in f[450:950]:
		f1.write(line+"\n")
	f1.close()


def similarity(v1,v2):
	return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def create_sim_matrix():
	l = ["50.txt","100.txt","300.txt","500.txt"]
	for f in l:
		sent_vecs = []
		lines = open(f).read()
		sents = lines.split("\n")
		words = map(word_tokenize,sents)
		model = Word2Vec(words, size=100, window=5, min_count=5, workers=4)
		final_sents = []
		for i in range(len(sents)):
			l=[]
			for word in word_tokenize(sents[i]):
				try:
					l.append(model[word])
				except:
					pass
			if len(l) > 0:
				sent_vecs.append([sents[i],reduce(np.add,l)])
				final_sents.append(sents[i])

		for i in range(len(sent_vecs)):
			sent_vecs[i][1] = sent_vecs[i][1]/np.linalg.norm(sent_vecs[i][1])
		
		table = {}

		for i in range(len(final_sents)):
			for j in range(len(final_sents)):
				if i != j:
					if final_sents[i] not in table.keys():
						table[final_sents[i]] = {}
						table[final_sents[i]][final_sents[j]] = similarity(sent_vecs[i][1],sent_vecs[j][1])
					else:
						table[final_sents[i]][final_sents[j]] = similarity(sent_vecs[i][1],sent_vecs[j][1])

		print "========================================"
		si = {}
		sb = {}
		for k,v in table.iteritems():
			si[k] = []
			sb[k] = []

			for item in v.items():
				temp = item[1]*10
				si[k].append((item[0],temp))
				if(temp <= 3):
					sb[k].append((item[0],0))
				else:
					sb[k].append((item[0],1))

		writer1 = csv.writer(open("Output_Si"+f.split(".")[0]+".csv","w"))
		writer2 = csv.writer(open("Output_Sb"+f.split(".")[0]+".csv","w"))
		

		for k,v in sorted(si.iteritems(), key=lambda key_value: key_value[0]):
			temp_si = []
			temp_si.append(k)
			t = list(set([item for item in v]))
			temp_si.extend(sorted(t))
			writer1.writerow(temp_si)

		for k,v in sorted(sb.iteritems(), key=lambda key_value: key_value[0]):
			temp_sb = []
			temp_sb.append(k)
			t = list(set([item for item in v]))
			temp_sb.extend(sorted(t))
			writer2.writerow(temp_sb)

		create_summary(sb,f)

def create_summary(sb,f):
	sums = {}
	for k,v in sb.iteritems():
		sums[k] = sum([item[1] for item in v])
	temp_l = sorted(sums,key=sums.get)[-1::-1][0]	# Take top 4 sentences 
	summary = []
	# for word in temp_l:
	item_list = sb[temp_l]
	summary.append([item[0] for item in item_list if item[1]==1])	# Add the words in the sentence with 
	urls = eval(open("urls.txt").read())
	for sent in summary:
		temp_sent = []
		for s in sent:
			for i in urls.keys():
				if i.find(s) > -1 or s.find(i)> -1: 
					print "HERE"
					s = re.sub(r'details',urls[i],s)
			temp_sent.append(s.strip(' '))
		open("Summary_"+f,"w").write("\n".join(temp_sent))

def create_html():
	for f in os.listdir("Summary"):
		fo = open("HTML_"+f,"w")
		lines = open("Summary/"+f).read().split("\n")
		for line in lines:
			if 'http'in line:
				res = re.search('(http[s]?:?/?/?((\w*[/]?\w*?)*(\.\w+)+([/]?\w+?)*)*)',line)
				url = res.group()
				line = line.replace(url,"")
				line = line.strip()
				new_line = "<a href='"+url+"'>"+line+"</a>"
			else:
				new_line = line.strip()
			output_line = "<li>"+new_line+"</li>"
			fo.write(str(output_line)+"\n")

# create_sim_matrix()
# create_html()