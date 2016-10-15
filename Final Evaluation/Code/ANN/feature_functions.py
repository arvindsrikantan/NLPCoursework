import nltk,re
from nltk.stem.porter import *

stemmer = PorterStemmer()


def f1(h):
	'''
	Event : Defended
	'''
	l = ['block','defend']
	for i in h:
		if i in l:
			return 1
	return 0

def f2(h):
	'''
	Event : Left alone
	'''
	l = ['lets','ducks','leaves','pulls','sways']
	for i in range(len(h)):
		if h[i] in l:
			if h[i] == 'lets':
				if 'go' in h[i:]:
					return 1
				else:
					return 0
			elif h[i] == 'pulls' or h[i] == 'sways':
				if 'out' in h[i:]:
					return 1
				else:
					return 0
			else:
				return 1
	return 0 

def f3(h):
	'''
	Event : Beaten
	'''
	l = ['miss','misses','beats','beaten','lbw','pad','doesn\'t']
	for i in h:
		if i in l:
			if i == 'doesn\'t':
				if 'connect' in h[h.index(i):]:
					return 1
				else:
					return 0
			else:
				return 1
	return 0

def f4(h):
	'''
	Event : Edged
	'''
	l = ['edged','edge']
	for i in h:
		if i in l:
			return 1
	return 0

def f5(h):
	'''
	Event : Caught
	'''
	for i in h:
		if i in ['caught','catches']:
			return 1
	return 0

def f6(h):
	'''
	Event : Runout
	'''
	for i in h:
		if i == 'OUT':
			if 'direct hit' in h[h.index(i):]:
				return 1
			else:
				return 0
		elif i == 'run':
			if 'out' in h[h.index(i):]:
				return 1
			else:
				return 0
	return 0

def f7(h):
	'''
	Event : Stumped
	'''
	if 'stumped' in h:
		return 1
	return 0

def f8(h):
	'''
	Event : Bowled
	'''
	if 'bowled' in h:
		return 1
	elif 'bails' in h:
		if 'off' in h[h.index('bails'):]:
			return 1
		else:
			return 0
	return 0

def f9(h):
	'''
	Event : LBW
	'''
	if 'lbw' in h:
		return 1
	return 0

def f10(h):
	'''
	Event : Boundary_scored_by_batsman
	'''
	if 'FOUR' in h:
		return 1
	elif 'SIX' in h:
		return 1
	return 0

def f11(h):
	'''
	Event : Runs_by_batsman
	'''
	nums = ['1','2','3']
	for i in h:
		if i in nums:
			if h[h.index(i)+1] == 'runs':
				return 1
			else:
				return 0
	return 0

def f12(h):
	'''
	Event : Boundary_scored_extras
	'''
	return 0

def f13(h):
	'''
	Event : Runs_by_extras
	'''
	if 'leg bye' in h:
		return 1
	return 0

def f14(h):
	'''
	Event : Catch_dropped
	'''
	for i in h:
		if i == 'catch':
			if 'drop' in h[h.index(i):]:
				return 1
			elif 'OUT' not in h[:h.index(i)]:
				return 1
			elif 'miss' in h[:h.index(i)]:
				return 1
		elif i == 'caught' and 'OUT' not in h[:h.index(i)]:
			return 1
	return 0

def f15(h):
	"""
	Event: Stumping Missed
	"""
	l = ["Stumps","Missed"]
	if(l[0] in h and l[1] in h):
		return 1
	else:
		return 0

def f16(h):
	"""
	Event: Runout Missed
	"""
	l = ["direct hit and missed","runout and missed"]
	if "direct" in h or "run out" in h:
		if("missed" in h):
			return 1
	else:
		return 0

def f17(h):
	"""
	Event: Bouncer
	"""
	if "bouncer" in h:
		return 1
	else:
		return 0

def f18(h):
	"""
	Event: Yorker
	"""
	if("yorker" in h or "block hole" in h):
		return 1
	else:
		return 0

def f19(h):
	"""
	Event: Overthrow
	"""
	if("extra" in h and "run" in h) or "overthrow" in h:
		return 1
	else:
		return 0

def f20(h):
	"""
	Event: great_save
	"""
	if(("chases" in h and "dive" in h) or ("runs" in h and "it in" in h) or ("runs" in h or "saves" in h) or ("great" in h or "save" in h)):
		return 1
	else:
		return 0

def f21(h):
	"""
	Event: poor_fielding
	"""
	if(("3 runs" in h) or ("poor" in h and "fielding" in h) or("fielder" in h or "fumbles" in h)):
		return 1
	else:
		return 0

def f22(h):
	"""
	Event: Free hit
	"""
	if("free hit" in h):
		return 1
	else:
		return 0