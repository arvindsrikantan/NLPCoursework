import copy,os
from nltk.tokenize import *
def create_triplets(inp):
	from nltk.stem import WordNetLemmatizer
	from nltk.stem.porter import PorterStemmer
	result = []
	stemmer=PorterStemmer()
	lemma=WordNetLemmatizer()
	inp=list(map(lemma.lemmatize,list(map(stemmer.stem,inp))))
	input = copy.deepcopy(inp)
	# input.insert(0,'*')
	input.insert(0,'*')
	input.append('*')
	
	for word in range(len(inp)):
		result.append((input[word: word+3]))
	'''print result
	import time
	time.sleep(5)'''
	return tuple(result)

def find_triplet_count(inp):
	result1 = {}
	result2 = {}
	unique_triplets = set(inp)
	
	for triplet in unique_triplets:
		result1[(triplet[0],triplet[2])] = {}
	
	for triplet in unique_triplets:
		result1[(triplet[0],triplet[2])][triplet[1]] = 0

	for triplet in unique_triplets:
		result1[(triplet[0],triplet[2])][triplet[1]] += 1
	f= open("model1.txt","w")
	f.write(str(result1))
	f.close()
	
	for triplet in unique_triplets:
		result2[triplet[1]] = {}
	
	for triplet in unique_triplets:
		result2[triplet[1]][(triplet[0], triplet[2])] = 0

	for triplet in unique_triplets:
		result2[triplet[1]][(triplet[0], triplet[2])] += 1
	f= open("model2.txt","w")
	f.write(str(result2))
	f.close()
	return result1, result2
	
def find_similarity(w1, w2, triplet1, triplet2):
	'''
		Find all common contexts, for the sum of all occurrences where the context
		is the same for the words w1 and w2. Normalize it with the count of all 
		triplets where w1 and w2 are the centres
	'''
	val_w1=triplet2[w1]
	val_w2=triplet2[w2]
	su = 0
	cont = 0
	for i in val_w1.keys():
		# print i
		if i in val_w2.keys():
			# print True
			su += val_w1[i] + val_w2[i]
			cont += sum(triplet1[i].values())
	
	count_w1 = sum(triplet2[w1].values())
	count_w2 = sum(triplet2[w2].values())
	
	# delta = abs(count_w1 - count_w2)
	Z_score = su / float(count_w1 + count_w2)
	other = su / float(cont)
	
	return Z_score,other

def main(inp,load=True):
	#print inp[0]
	

	if not load:
		l = list(map(create_triplets, [i for i in inp if len(i)>1]))
		li=[]
		for i in l:
			for j in i:
				li.append(tuple(j))
		#print type(li[0]), type(li[0][0])
		model1, model2 = find_triplet_count(li)
	else:
		model1 = inp[0]
		model2 = inp[1]
		
	while True:
		w1=raw_input("Enter w1:")
		w2=raw_input("Enter w2:")
		print "Similarity(Z_score) =", find_similarity(w1,w2, model1, model2)
		print "Word2Vec =", W2VCompare(w1,w2)

def train_word2vec():
	from sentence2vec.word2vec import Word2Vec, Sent2Vec, LineSentence
	tweets = tc.cleaner(open("tweets.txt").read().lower())
	sent = sent_tokenize(tweets)
	words = map(word_tokenize,sent)
	global model
	model=Word2Vec(words, size=8, window=3, min_count=5, workers=16)

def W2VCompare(w1,w2):
	return (model.similarity(w1,w2)+1)/2.0
	
if __name__ == "__main__":
	import TweetCleaner as tc
	import nltk
	train_word2vec()
	if(not os.path.exists("model1.txt")):
		clean_tweets = list(map(tc.cleaner, list(map(str.lower, open("tweets.txt").readlines()))))
		main(list(map(nltk.word_tokenize, clean_tweets)), load = False)
	else:
		main([eval(open("model1.txt").read()), eval(open("model2.txt").read())])
	