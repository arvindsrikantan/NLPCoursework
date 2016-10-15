import nltk, os
from sentence2vec.word2vec import Word2Vec, Sent2Vec, LineSentence
wordList = []#list(map(nltk.word_tokenize, nltk.sent_tokenize(open("datasets/Abdul Kalam.txt").read().lower())))
for root, dirs, files in os.walk("datasets"):
	for file in files:
		wordList.extend(list(map(nltk.word_tokenize, nltk.sent_tokenize(open(root+"/"+file).read().lower()))))
#print wordList[:2]
input_file = 'test.txt'
model = Word2Vec(wordList, size=10, window=20, sg=0, min_count=5, workers=8)
model.save(input_file + '.model')
model.save_word2vec_format(input_file + '.vec')
model = Word2Vec.load(input_file + '.model')
while True:
	try:
		w1 = raw_input("Enter w1\n").lower()
		w2 = raw_input("Enter w2\n").lower()
		# print w1,w2
		# print model[w1], "  ",model[w2]
		print "similarity =",model.similarity(w1,w2)
		print "*"*25,"\n"*2
		
	except Exception as ex:
		print ex.args
		print ex.message
		#help(ex)