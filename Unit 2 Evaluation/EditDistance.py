
def editDistance(s1,s2):
	s1 = " " + s1
	s2 = " " + s2
	M = len(s1)
	N = len(s2)
	d = [[0 for i in range(M)] for j in range(N)]
	
	for i in range(M):
		d[0][i] = i

	for i in range(N):
		d[i][0] = i
		
	for i in range(1,N):
		for j in range(1,M):
			d[i][j] = min( (d[i-1][j]+1),(d[i][j-1]+1),((d[i-1][j-1]+2) if not s2[i] == s1[j] else d[i-1][j-1]) )
			
	# for i in range(N):
		# for j in range(M):
			# print d[i][j],
		# print ""
		
	print "Edit distance = ", d[N-1][M-1]," for strings :",s1,s2
	return d[N-1][M-1]

if __name__ == "__main__":
	s2 = raw_input("Enter source word: ")
	s1 = raw_input("Enter target word: ")
	editDistance(s1, s2)
	s2 = raw_input("Enter source word: ")
	s1 = eval(raw_input("Enter target list: "))
	di = {}
	for i in s1:
		di[i] = editDistance(s2, i)
	print di
	print "Closest word = ", di.keys()[di.values().index(min(di.values()))]
	