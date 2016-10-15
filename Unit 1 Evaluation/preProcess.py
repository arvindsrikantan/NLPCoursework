from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

def con(l):
	fo=open("raw_data.txt","w");
	ol=[]
	for fi in l:
		f=open(fi,"r")
		ol.extend(eval(f.read()))
		f.close()
	fo.write(str(ol))
	fo.close()

def createTokens(f, file = True):
	if file:
		text=eval(open(f,"r").read())
	else:
		text = f
	text=list(map(str.lower,[i for i in text]))
	# words=map(word_tokenize,text)
	words=[]
	for i in text:
		words.append(re.split(r"\W+",i))
	#stop = stopwords.words('english')
	stop = ["if","then","and","or","is","are","the","a"]
	new_word=[]
	for i in words:
		l=[]
		for j in i:
			if(j not in stop):
				l.append(j)
		new_word.append(l)
	final_word=[]
	for i in new_word:
		l=[]
		for j in i:
			j=re.sub("4|6|four|six","boundary",j)
			j=re.sub("couple|2|double","two",j)
			j=re.sub("single|1","one",j)
			j=re.sub("3","three",j)
			l.append(j)
		final_word.append(l)
	return final_word


	
def lemmatize(s):
	wordnet_lemmatizer = WordNetLemmatizer()
	new_list = []
	for i in s:
		l=[]
		for j in i:
			l.append(wordnet_lemmatizer.lemmatize(j))
		new_list.append(l)
	return new_list

def stemmer(s):
	porter_stemmer = PorterStemmer()
	new_list = []
	for i in s:
		l=[]
		for j in i:
			l.append(porter_stemmer.stem(j))
		new_list.append(l)
	return new_list


if __name__=="__main__":
	#con(["Linear-1.txt","Linear-2.txt","Linear-3.txt","Linear-4.txt","Linear-5.txt"])
	tokens=createTokens("raw_data.txt")
	tokens=stemmer(tokens)
	tokens=lemmatize(tokens)
	open("corpus.txt","w").write(str(tokens))


