import copy, math
def get_vocab(docs):
	vocab = []
	for i in docs:
		vocab.extend(i)
	return list(set(vocab))


def calculate_tf(docs):
	global d
	for i in docs:
		temp = copy.deepcopy(d)
		for word in i:
			temp[word] += 1
		l.append(temp)
	out = open("tf.txt", "w")
	out.write(str(l))

def calculate_idf(docs):
	global d
	temp = copy.deepcopy(d)
	
	for t in V:
		d_count = 0
		for d in docs:		
			if t in d:
				d_count += 1
		temp[t] = math.log(len(docs)/d_count)
	out = open("idf.txt", "w")
	out.write(str(temp))

def load_idf():
	return eval(open("idf.txt").read())

def load_tf():
	return eval(open("tf.txt").read())

def calculate_tfidf_OR(tf, idf, inp, docs):
	document = 0
	score = []
	for i in range(len(tf)):
		for word in inp:
			# print i, word
			# print idf[word]
			# print tf[i][word]
			
			try:
				document += tf[i][word] * idf[word]
			except:
				pass
		score.append((document, docs[i]))
	return sorted(score, reverse = True, key = lambda x:x[0])

def calculate_tfidf_AND(tf, idf, inp, docs):
	document = 0
	score = []
	for i in range(len(tf)):
		flag = True
		for word in inp:
			try:
				if(tf[i][word] == 0):
					flag = False
					# print "Jai " * 80
					break
			except:
				pass
		if flag:
			for word in inp:
				try:
					document += tf[i][word] * idf[word]
				except:
					pass
		else:
			document = 0
		score.append((document, docs[i]))
	return sorted(score, reverse = True, key = lambda x:x[0])



if __name__ == '__main__':
	l=[]
	docs = eval(open("corpus.txt").read())
	documents = eval(open("raw_data.txt").read())
	d = {}
	V = get_vocab(docs)
	ff = open("output.txt","w")
	for i in V:
		d[i] = 0
	
	# calculate_tf(docs)
	# calculate_idf(docs)
	
	tf = load_tf()
	idf = load_idf()
	# print len(tf), type(idf), type(tf[0])
	import preProcess as pp
	while True:
		inp = raw_input("Enter query:")
		f = True #Or
		if '"' == inp[0]:
			inp = inp[1:-1]
			f = False
		tokens = pp.createTokens([inp], file = False)
		tokens = pp.stemmer(tokens)
		tokens = pp.lemmatize(tokens)
		print tokens
		res = []
		if f:
			print "Logical OR query:"
			res = calculate_tfidf_OR(tf, idf, tokens[0], documents)
		else:
			print "Logical AND query:"
			res = calculate_tfidf_AND(tf, idf, tokens[0], documents)
		if res[0][0] == 0:
			print "None of the search terms are present in the list of documents"
			ff.write("Input : "+inp+"\nOutput : None of the search terms are present in the list of documents\n")
		else:
			print "Result =", [i for i in res[:10] if i[0]>0]
			ff.write("Input : "+inp+"\nOutput : "+str([i for i in res[:10] if i[0]>0])+"\n")