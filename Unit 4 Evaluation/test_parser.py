"""
test_parser.py - A simple Python client for invoking Stanford CoreNLP Parser
Author: Anantharaman Narayana Iyer
Pre requisites to use this client:
    - JDK 1.8 environment
    - Stanford Parser installed
"""
import os
from nltk.parse import stanford
from nltk import sent_tokenize
os.environ['STANFORD_PARSER'] = r"D:\engineering\CSE 7th sem\NLP\Unit eval 4\stanford\jars"
os.environ['STANFORD_MODELS'] = r"D:\engineering\CSE 7th sem\NLP\Unit eval 4\stanford\jars"
os.environ['JAVAHOME'] = r"C:\Program Files\Java\jdk1.8.0_45\bin"
parser = stanford.StanfordParser()
f = open("trees.txt","w")
l = []
while True:
    # txt = raw_input("Enter text to be parsed, QUIT for quit: ")
    txt = open("anaphora_dataset.txt").read()
    if txt == "QUIT":
        break
    sentences = parser.raw_parse_sents(sent_tokenize(txt))
    # sentences = parser.raw_parse_sents(txt)
    for line in sentences:
            for sentence in line:
                    l.append(sentence)
                    #sentence.draw() 					
    # print sentences
    f.write(str(l))
    break
