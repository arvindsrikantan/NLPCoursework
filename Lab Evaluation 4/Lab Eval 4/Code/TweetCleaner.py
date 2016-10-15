import re
from nltk.tokenize import *
from gensim.models import *
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import time
from nltk.corpus import stopwords


# [HTML tags, handle, URL, Hash tags, punctuation]
regex1= [r'<[^>]+>' , r'(?:@\w+)', r'\b\w{1}\b', r'(?:\#+[\w]+[\w\'\-]*[\w]+)', r'[:,+;$%|\'/]', r'^RT\b|rt\]]b', r'/\w+']
# Underscore and hyphen within words
regex2 = [r'[_\-@&$''""?]']
stop = stopwords.words('english')
urls = dict()
def cleaner(s):
	s.lower()
	url = ""
	flag = 0
	s = re.sub(r':[PD)(] | ;[PD)(]',r'',s)	# Emoticons
	if 'http' in s:
		res = re.search('(http[s]?:?/?/?((\w*[/]?\w*?)*(\.\w+)+([/]?\w+?)*)*)',s)
		if res is not None:
			s = re.sub('http[s]?:?/?/?((\w*[/]?\w*?)*(\.\w+)+([/]?\w+?)*)*',r'details',s)
			flag = 1
			url = res.group()
			

	for r in regex1:
			s = re.sub(r,r'',s)
	for r in regex2:
		s = re.sub(r,r'',s)
	# s = re.sub(r'\s+',r' ',s)

	s = s.lower()
	if flag:
		urls[s.lower()] = url

	return s


def W2VTest(s,w1,w2):
	l1 = sent_tokenize(s)
	l2 = map(word_tokenize,l1)
	model = Word2Vec(l2, size=8, window=5, min_count=5, workers=4)
	print(model.similarity(w1,w2))

def lemmatize(s):
	wordnet_lemmatizer = WordNetLemmatizer()
	l = word_tokenize(s)
	new_list = []
	for i in range(len(l)):
		if l[i] not in stop: 
			new_list.append(wordnet_lemmatizer.lemmatize(l[i]))
	return " ".join(new_list)

def stemmer(s):
	porter_stemmer = PorterStemmer()
	l = word_tokenize(s)
	new_list = []
	for i in range(len(l)):
		if l[i] not in stop: 
			new_list.append(porter_stemmer.stem(l[i]))
	return " ".join(new_list)

#cleaner("RT @marcobonzanini: just.an example_s! :D http://exam_ple.com #NLP#NLP")
#lemmatize("superficially resembles a churches")
f=open("tweets.txt","r")
f1 = open("cleanedTweets.txt","w")
final=[]
for line in f:
	sent = cleaner(line)
	# sent = stemmer(sent)
	# sent = lemmatize(sent)
	final.append(sent)

for line in final:
	if(len(line) > 1):
		f1.write(str(line).lower())
f1.close()
open("urls.txt","w").write(str(urls))
#stemmer("superficially resembles a churches")