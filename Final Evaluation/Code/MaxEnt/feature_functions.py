import nltk,re
from nltk.stem.porter import *

stemmer = PorterStemmer()


def f1(h,t):
	'''
	Event : Defended
	'''
	l = ['block','defend']
	if t == 'Defended':
		for i in h:
			if i in l:
				return 1
	return 0

def f2(h,t):
	'''
	Event : Left alone
	'''
	l = ['lets','ducks','leaves','pulls','sways']
	if t == 'Left alone':
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

def f3(h,t):
	'''
	Event : Beaten
	'''
	l = ['miss','misses','beats','beaten','lbw','pad','doesn\'t']
	if t == 'Beaten':
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

def f4(h,t):
	'''
	Event : Edged
	'''
	l = ['edged','edge']
	if t == 'Edged':
		for i in h:
			if i in l:
				return 1
	return 0

def f5(h,t):
	'''
	Event : Caught
	'''
	if t == 'Caught':
		for i in h:
			if i in ['caught','catches']:
				return 1
	return 0

def f6(h,t):
	'''
	Event : Runout
	'''
	if t == 'Runout':
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

def f7(h,t):
	'''
	Event : Stumped
	'''
	if 'stumped' in h and t == 'Stumped':
		return 1
	return 0

def f8(h,t):
	'''
	Event : Bowled
	'''
	if t == 'Bowled':
		if 'bowled' in h:
			return 1
		elif 'bails' in h:
			if 'off' in h[h.index('bails'):]:
				return 1
			else:
				return 0
	return 0

def f9(h,t):
	'''
	Event : LBW
	'''
	if 'lbw' in h and t == 'LBW':
		return 1
	return 0

def f10(h,t):
	'''
	Event : Boundary_scored_by_batsman
	'''
	if 'FOUR' in h and t == 'Boundary_scored_by_batsman':
		return 1
	elif 'SIX' in h and t == 'Boundary_scored_by_batsman':
		return 1
	return 0

def f11(h,t):
	'''
	Event : Runs_by_batsman
	'''
	nums = ['1','2','3']
	if t == 'Runs_by_batsman':
		for i in h:
			if i in nums:
				if h[h.index(i)+1] == 'runs':
					return 1
				else:
					return 0
	return 0

def f12(h,t):
	'''
	Event : Boundary_scored_extras
	'''
	return 0

def f13(h,t):
	'''
	Event : Runs_by_extras
	'''
	if 'leg bye' in h and t == 'Runs_by_extras':
		return 1
	return 0

def f14(h,t):
	'''
	Event : Catch_dropped
	'''
	if t == 'Catch_dropped':
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

def f15(h,t):
	"""
	Event: Stumping Missed
	"""
	l = ["Stumps","Missed"]
	if(l[0] in h and l[1] in h and t == 'Stumping Missed'):
		return 1
	else:
		return 0

def f16(h,t):
	"""
	Event: Runout Missed
	"""
	l = ["direct hit and missed","runout and missed"]
	if "direct" in h or "run out" in h and t == 'Runout Missed':
		if("missed" in h):
			return 1
	else:
		return 0

def f17(h,t):
	"""
	Event: Bouncer
	"""
	if "bouncer" in h and t == 'Bouncer':
		return 1
	else:
		return 0

def f18(h,t):
	"""
	Event: Yorker
	"""
	if("yorker" in h or "block hole" in h and t == 'Yorker'):
		return 1
	else:
		return 0

def f19(h,t):
	"""
	Event: Overthrow
	"""
	if("extra" in h and "run" in h) or "overthrow" in h and t == 'Overthrow':
		return 1
	else:
		return 0

def f20(h,t):
	"""
	Event: great_save
	"""
	if(("chases" in h and "dive" in h) or ("runs" in h and "it in" in h) or ("runs" in h or "saves" in h) or ("great" in h or "save" in h) and t == 'great_save'):
		return 1
	else:
		return 0

def f21(h,t):
	"""
	Event: poor_fielding
	"""
	if(("3 runs" in h) or ("poor" in h and "fielding" in h) or("fielder" in h or "fumbles" in h) and t == 'poor_fielding'):
		return 1
	else:
		return 0

def f22(h,t):
	"""
	Event: Free hit
	"""
	if "free hit" in h and t == 'Free hit':
		return 1
	else:
		return 0