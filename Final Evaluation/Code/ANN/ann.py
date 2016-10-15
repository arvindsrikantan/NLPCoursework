import neurolab as nl
import numpy as np
import time
import random
from event_tagger import *

def threshold(l, t = 0.4):
	temp = []
	for i in l:
		if(i<t):
			temp.append(0)
		else:
			temp.append(1)
	return temp

net = nl.net.newff([(-1,1)] * 22, [16, 22])
err = net.train(inputs, targets, show=1,epochs=10000)
net.save("ann.pickle")
net = nl.load("ann.pickle")
for i in range(len(inputs[2:10])):
	print threshold(net.sim([inputs[i]])[0].tolist()), targets[i]
