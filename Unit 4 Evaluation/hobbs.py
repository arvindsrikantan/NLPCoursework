import nltk
from nltk.tree import *
import os,copy
from nltk.tokenize import sent_tokenize
p=[]
def bfs(node, label):
	bfs_l = [node]
	foundNode = False
	while len(bfs_l)>0:
		temp = bfs_l.pop(0)
		if(not type(temp)==str):
			for child in temp:
				bfs_l.append(child)
				# print child.label() if not type(child) == str else child
				# print child.label()
				if((not type(child) == str) and child.label() == label):
					print "Found" #break here from bfs
					return child
					foundNode = True
					break
			if foundNode == True:
				break

def dfs(tree1,node):
	#print("tree:", tree)
	for subtree in tree1:
		if not type(subtree) == str:
			if(not subtree==node):
				#print subtree.label()
				p.append(subtree)
				dfs(subtree,node)
				p.pop()
			else:
				#print type(p),"p:",p
				return p

def findHighestS(tree1):
	return bfs(tree1,"S")
				
def findpath(node1, node2):
	p=[]
	return dfs(node1, node2)

def propose(node,pro):
	#handle singularity, plurality
	l=[]
	for i in node.leaves():
		for subtree in node.subtrees():
		#	print subtree[0]
			if(subtree[0] == i):
				p_sub1 = subtree
				l.append(p_sub1.label())
	#print l
	if pro in ['they','their',"their's",'our','ours','these']:
		if 'NNS' in l:
			return True
		return False
	if pro in ['it','its','he','she','his','her','hers','this']:
		if 'NN' in l:
			return True
		return False
	return True
	#

def hobbs(pro):
	global tree
	for subtree in tree.subtrees():
	#	print subtree[0]
		if(subtree[0] == pro):
			p_sub = subtree
			
	
	from nltk.parse import stanford
	from nltk import sent_tokenize
	os.environ['STANFORD_PARSER'] = r"D:\engineering\CSE 7th sem\NLP\Unit eval 4\stanford\jars"
	os.environ['STANFORD_MODELS'] = r"D:\engineering\CSE 7th sem\NLP\Unit eval 4\stanford\jars"
	os.environ['JAVAHOME'] = r"C:\Program Files\Java\jdk1.8.0_45\bin"
	parser = stanford.StanfordParser()
	#print p_sub
	temp = p_sub.parent()
	left=[p_sub.left_sibling()]
	right=[p_sub.right_sibling()]
	path=[temp]
	#print temp.left_sibling().left_sibling()
	#tree.draw()
	while not(temp.label() == 'NP' or temp.label() == 'S'):
		path.append(temp)
		temp = temp.parent()
	X=temp
	#print X
	while True:
		#2a
		#print "Step 2 a"
		path.append(X)
		temp = X.parent()
		while not(temp.label() == 'NP' or temp.label() == 'S'):
			path.append(temp)
			temp = temp.parent()
		X=temp
		#print X
		proposed = False
		found = False
		#2b
		#print "Step 2 b"
		'''bfs_l = [X]
		#print X
		if X.label()=="NP" or X.label() == "S":
			found = True
		while len(bfs_l)>0 or proposed == False:
			cc =0
			mm = bfs_l.pop()
			print mm
			import time
			for st in mm:
				ttt = st
				break
			
			while not ttt == None:
				#print "ttttt",ttt,type(ttt),"\n","bfs",bfs_l,"mm",mm
				if ttt not in path:
					bfs_l.append(ttt)
					print bfs_l,ttt
					time.sleep(3)
					cc += 1
					if (not type(ttt)==str) and ttt.label() == 'NP' and found == True:
						#propose
						X = ttt
						proposed = propose(X,pro)
						print X
						break
					if (not type(ttt)==str) and (ttt.label() == 'NP' or ttt.label() == 'S'):
						found = True
					if (not type(ttt)==str):
						ttt = ttt.right_sibling()
					if type(ttt) == str:
						#bfs_l.pop()
						break
				else:
					break
			
			for i in range(cc-1):
				bfs_l.pop()'''
		if X.label()=="NP" or X.label() == "S":
			found = True
		bfs_l = [X]
		bool = False
		foundNode = False
		tt=0
		while len(bfs_l)>0:
			temp = bfs_l.pop(0)
			if(not type(temp)==str):
				for child in temp:
					bfs_l.append(child)
					# print child.label() if not type(child) == str else child
					# print child.label()
					if(child not in path) and (not type(child) == str) and child.label() == 'NP' and found == True:
						# print "Found", type(findpath(X,child)) #break here from bfs
						path.append(child)
						tt = child
						proposed = propose(child,pro)
						foundNode = True
						break
					if child in path:
						bool = True
				if foundNode == True or bool:
					break
		#print "outside while"
		#print path
		if proposed == True:
			print tt,"proposed"
			break
		else:
			print tt,"Not proposed"
			
		#print "Step 3"
		#step 3
		#print "Highest", findHighestS(tree)
		if X==findHighestS(tree):
			pp = ll[ll.index(tree)-1]
			tt=0
			print "previous tree", "*"*100
			bfs_l = [pp]
			foundNode = False
			while len(bfs_l)>0:
				temp = bfs_l.pop(0)
				if(not type(temp)==str):
					for child in temp:
						bfs_l.append(child)
						# print child.label() if not type(child) == str else child
						# print child.label()
						if((not type(child) == str) and child.label() == "NP"):
							print "Found" #break here from bfs
							tt=child
							proposed = propose(child,pro)
							foundNode = True
							break
					if foundNode == True:
						break
			if proposed == True:
				print tt,"proposed"
				break
			else:
				print tt,"Not proposed"
			
			# break
		
		#step 4
		else:
			#print "Step 4"
			temp = X.parent()
			while not(temp.label() == 'NP' or temp.label() == 'S'):
				path.append(temp)
				temp = temp.parent()
				X = temp
		#print "Step 5"
		#step 5
		# no nbar so chill
		
		
		#step 6
		#print "Step 6"
		bfs_l = [X]
		proposed = False
		'''while len(bfs_l)>0 or proposed == False:
			cc =0
			mm = bfs_l.pop()
			for st in mm:
				ttt = st
				break
			while not ttt == None:
				if ttt not in path:
					bfs_l.append(ttt)
					cc += 1
					if (not type(ttt)==str) and ttt.label() == 'NP':
						#propose
						X = ttt
						print X
						proposed = propose(X,pro)
						break
					if (not type(ttt)==str):
						ttt = ttt.right_sibling()
					else:
						break
				else:
					break
			for i in range(cc-1):
				bfs_l.pop()'''
		bool = True
		pp=0
		while len(bfs_l)>0:
			temp = bfs_l.pop(0)
			if(not type(temp)==str):
				for child in temp:
					bfs_l.append(child)
					# print child.label() if not type(child) == str else child
					# print child.label()
					if(child not in path) and (not type(child) == str) and child.label() == 'NP':
						print "Found 6" #break here from bfs
						# path.append(X)
						pp = child
						proposed = propose(pp,pro)
						foundNode = True
						break
					if child in path:
						bool = True
				if foundNode == True or bool:
					break
		for i in X:
			X=i
			break
		#print "outside while2"
		
		if proposed == True:
			print pp,"proposed"
			break
		else:
			print pp,"Not proposed 6"
		
		#step 7
		#print "Step 7"
		if(X.label()=="S"):
			bfs_l = [X]
			proposed = False
			'''while len(bfs_l)>0 or proposed == False:
				cc =0
				mm = bfs_l.pop()
				for st in mm:
					ttt = st
					break
				while not ttt == None:
					if ttt not in path:
						bfs_l.append(ttt)
						cc += 1
						if (not type(ttt)==str) and ttt.label() == 'NP':
							#propose
							X = ttt
							print X
							proposed = propose(X,pro)
							break
						if (not type(ttt)==str):
							ttt = ttt.right_sibling()
						else:
							break
					else:
						break
				for i in range(cc-1):
					bfs_l.pop()'''
			import time
			bool = False
			pp=0
			while len(bfs_l)>0:
				temp = bfs_l.pop(0)
				if(not type(temp)==str):
					for child in temp:
						if child in path:
							continue
						bfs_l.append(child)
						# print child.label() if not type(child) == str else child
						# print child.label()
						if(not type(child) == str) and child.label() == 'S':
							bool = True
							print "breaking"
							time.sleep(3)
							break
						elif(not type(child) == str) and child.label() == 'NP':
							print "Found" #break here from bfs
							# path.append(X)
							pp = child
							proposed = propose(pp,pro)
							print pp
							foundNode = True
							time.sleep(4)
							break
						if child in path:
							bool = True
				if foundNode == True or bool:
					break
					
			#print "outside while3"
			if not bool and proposed == True:
				print pp,"proposed"
				break
			else:
				print pp,"Not proposed"
		temp = X
		# break
	
	
	#print temp,"=======",temp.label()
s = open('anaphora_dataset.txt','r').read()
ll = list(map(ParentedTree.convert,eval(open("trees.txt").read())))
tree = ll[-1]
print '********************************'
print 'Input Sentence : ',sent_tokenize(s)[5]
print 'Pronoun : that'
print 'Node Proposed : '
#print tree
tree.draw()
#bfs(tree, "NP")	
hobbs('their')
print '********************************'
