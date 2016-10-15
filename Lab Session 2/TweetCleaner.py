import re
from nltk.tokenize import *
from gensim.models import *
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


# [HTML tags, handle, URL, Hash tags, punctuation]
regex1= [r'<[^>]+>' , r'(?:@\w+)', r'http[s]?:?/?/?((\w+[/]?\w*?)*(\.\w+)+([/]?\w+?)*)*', r'(?:\#+[\w]+[\w\'\-]*[\w]+)', r'[:,+;.!/]', r'^RT\b', r'/\w+']
# Underscore and hyphen within words
regex2 = [r'[_\-]']

def cleaner(s):
	s.lower()
	s = re.sub(r':[PD)(] | ;[PD)(]',r'',s)	# Emoticons
	for r in regex1:
		s = re.sub(r,r' ',s)
	for r in regex2:
		s = re.sub(r,r'',s)
	s.strip()
	s = re.sub(r'\s+',r' ',s)

	s.lower()
	# print(s)
	return s


def W2VTest(s,w1,w2):
	l1 = sent_tokenize(s)
	l2 = map(word_tokenize,l1)
	model = Word2Vec(l2, size=8, window=5, min_count=5, workers=4)
	print(model.similarity(w1,w2))

def lemmatize(s):
	wordnet_lemmatizer = WordNetLemmatizer()
	l1 = sent_tokenize(s)
	l2 = map(word_tokenize,l1)
	print(l2)
	new_list = []
	for i in range(len(l2)):
		new_list.append([])
		for j in range(len(l2[i])):
			new_list[i].append(wordnet_lemmatizer.lemmatize(l2[i][j]))
	print(new_list)

def stemmer(s):
	porter_stemmer = PorterStemmer()
	l1 = sent_tokenize(s)
	l2 = map(word_tokenize,l1)
	print(l2)
	new_list = []
	for i in range(len(l2)):
		new_list.append([])
		for j in range(len(l2[i])):
			new_list[i].append(porter_stemmer.stem((l2[i][j])))
	print(new_list)
if __name__ == "__main__":
	#cleaner("RT @marcobonzanini: just.an example_s! :D http://exam_ple.com #NLP#NLP")
	#lemmatize("superficially resembles a churches")
	f=open("tweets.txt","r")
	final=[]
	for line in f:
		final.append(cleaner(line))
	for line in final:
		print(line)
	#stemmer("superficially resembles a churches")