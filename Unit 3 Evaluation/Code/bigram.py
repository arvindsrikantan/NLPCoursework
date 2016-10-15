import math
from nltk.tokenize import *

def prepare_LM(inp_file,target_file):
	d = {}
	sents = sent_tokenize(open(inp_file).read().strip())
	for sent in sents:
		words = word_tokenize(sent)
		words.insert(0,'*')

		#get the unigram counts
		for i in words:
			if i in d.keys():
				d[i] += 1
			else:
				d[i] = 1

		#get all bigram counts
		for  i in range(len(words)-1):
			if (words[i],words[i+1]) in d.keys():
				d[(words[i],words[i+1])] += 1
			else:
				d[(words[i],words[i+1])] = 1

	#prepare the bigram LM
	for i in d.keys():
		if type(i) == tuple:
			d[i] = d[i]/float(d[i[0]])

	open(target_file,'w').write(str(d))

def predict_prob(sent,target_file):
	d = eval(open(target_file).read())
	words = word_tokenize(sent)
	words.insert(0,'*')
	l = [(words[i],words[i+1]) for i in range(len(words)-1)]
	prob = 1
	for i in l:
		if i in d.keys():
			prob += math.log(d[i])
		else:
			#discount probability if not found
			prob +=  math.log(10 ** -4)
			# pass
	#print d
	return prob

def perplexity(sent,target_file):
	# print predict_prob(sent,target_file)
	return 2 ** -(predict_prob(sent,target_file)/float(len(word_tokenize(sent))))

if __name__=="__main__":
	prepare_LM('input.txt','wiki.model')
	# print predict("dhyan chand")
	# print perplexity("dhyan chand")