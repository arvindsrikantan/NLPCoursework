"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
from numpy import array
import csv,os
from gensim.models import *
import re
from nltk.tokenize import *


data = eval(open("cleaned_tags.txt").read())
vocab = list(set(zip(*data)[0]))

tags = ["batsman","non-striker","bowler","fielder","fielding_position","other"] #list(set(data))
data_size, vocab_size = len(data), 16
# words = re.sub(r'(.+), ',r'\1 , ',str.lower(open("all_comms.txt").read())).split("\n")
# sentences = [re.split(r' ',s) for s in words]
l = sent_tokenize(str.lower(open("all_comms.txt").read()))
sentences = map(word_tokenize,l)

model = Word2Vec(sentences, size=16, window=5, min_count=0, workers=4)
# open("model.txt","w").write(str(model))

# print 'data has %d characters, %d unique.' % (data_size, vocab_size)
word_to_ix =  {}  # make it word:word2vec vector 16D
for sent in sentences:
	for word in sent:
	  try:
	    word_to_ix[word] = model[word]
	  except:
	    print word

# ix_to_word = { i:ch for i,ch in enumerate(vocab) }
tag_to_ix = { ch:i for i,ch in enumerate(tags) }
ix_to_tag = { i:ch for i,ch in enumerate(tags) }

# hyperparameters
hidden_size = 8 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(6, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((6, 1)) # output bias

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t] = np.reshape(inputs[t],(16,1)) # shaaaaaaaaaaady ------------ xs[t] = wordvec.Transpose  ---  Train the Word2Vec for a word representation of dimension 16 valued vector
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y
 
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

index =  0
# while n<=1000:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  # if p+seq_length+1 >= len(data) or n == 0 or index+25 >= len(data): 
    # hprev = np.zeros((hidden_size,1)) # reset RNN memory
    # p = 0 # go from start of data
  
  # inputs = []
  # counter = 0
  # while len(inputs) < 25 :
	# try:
  		# inputs.append(word_to_ix[zip(*data)[0][index+counter]])
  		# counter += 1
  	# except:
  		# counter += 1
  # index += 25

  # targets = [tag_to_ix[ch] for ch in zip(*data)[1][p:p+seq_length]]
  # lossFun(inputs, targets, hprev)
  # sample from the model now and then
  # if n % 100 == 0:
    # sample_ix = sample(hprev, inputs[0], 200)
    # txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    # print '----\n %s \n----' % (txt, )

  # forward seq_length characters through the net and fetch gradient
  # loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)

  # smooth_loss = smooth_loss * 0.999 + loss * 0.001
  # if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
  
  # perform parameter update with Adagrad
  # for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                # [dWxh, dWhh, dWhy, dbh, dby], 
                                # [mWxh, mWhh, mWhy, mbh, mby]):
    # mem += dparam * dparam
    # param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  # p += seq_length # move data pointer
  # n += 1 # iteration counter 

# open("vectors.p","w").write(str([dWxh, dWhh, dWhy, dbh, dby, hprev]))
dWxh, dWhh, dWhy, dbh, dby, hprev = eval(open("vectors.p","r").read())

'''
Testing
'''
print "Starting test"
p=0
match_count = 0
while p < len(vocab)-len(vocab)%25:
	inputs = []
	abc = zip(*data)[0][p:p+seq_length]
	for w in abc:
		if w in word_to_ix.keys():
			inputs.append(w)
	 #check for first
	targets = [tag_to_ix[ch] for ch in zip(*data)[1][p:p+seq_length]]
	xs, hs, ys, ps = {}, {}, {}, {}
	hprev = np.zeros((hidden_size,1))
	hs[-1] = np.copy(hprev)
	for t in xrange(len(inputs)):
		xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
		xs[t] = np.reshape(word_to_ix[inputs[t]],(16,1))

		hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
		ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
		ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
		index = ps[t].flatten().tolist().index(max(ps[t].flatten())),
		observed = ix_to_tag[ps[t].flatten().tolist().index(max(ps[t].flatten()))]
		print "observed : ",observed, " target : ",targets[t]
		if index == targets[t]:
			match_count += 1
print match_count/float(len(vocab))