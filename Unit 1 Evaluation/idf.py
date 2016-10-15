def idf(t,D):
	d_count = 0
	for each d in D:
		if t in d:
			d_count += 1
	return math.log(len(D)/d_count)
