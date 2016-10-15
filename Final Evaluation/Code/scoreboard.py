from html import HTML
import re
d = eval(open('Match_4_2.json').read()) #keys: comments, bowlers, outs
'''
	<batsmen> \s c \s <by> \s b \s <bowler> \s runs(Xm \s Xb \s Xx4 \s Xx6) \s SR: \s num
	<batsmen> \s run \s out \s <bowler> \s runs(Xm \s Xb \s Xx4 \s Xx6) \s SR: \s num
	<batsmen> \s lbw \s b \s <bowler> \s runs(Xm \s Xb \s Xx4 \s Xx6) \s SR: \s num	
	<batsmen> \s b \s <bowler> \s runs(Xm \s Xb \s Xx4 \s Xx6) \s SR: \s num
'''
h = HTML()
t = h.table(border='1')
r=t.tr
r.th("Batsmen")
r.th("Batting status")
r.th("R")
r.th("M")
r.th("B")
r.th("4s")
r.th("6s")
r.th("SR")
for i in d["outs"]:
	mat = re.search(r"([0-9]+) \(([0-9]+)m ([0-9]+)b ([0-9]+)x4 ([0-9]+)x6\) SR: ([0-9]+\.[0-9]+)",i)
	if(mat == None):
		continue
	r = t.tr
	if(" c " in i and " b " in i):
		r.td(i.split(" c ")[0])
		r.td("c "+re.split(r"[0-9]+ \(",i.split(" c ")[1])[0])
	elif(" run out " in i):
		r.td(i.split(" run ")[0])
		r.td("run out "+re.split(r"[0-9]+ \(",i.split(" run out ")[1])[0]+"(Unknown)")
	elif(" lbw " in i):
		r.td(i.split(" lbw ")[0])
		r.td("lbw "+re.split(r"[0-9]+ \(",i.split(" lbw ")[1])[0])
	else:
		r.td(i.split(" b ")[0])
		#print i.split(" b ")
		r.td("b "+re.split(r"[0-9]+ \(",i.split(" b ")[1])[0])
	
	#print mat,i
	
	for j in range(1,7):
		r.td(mat.group(j))
		
'''
	Extras:
		(no \s ball) \s (FOUR|SIX|num)  -------- Means add
		1 no ball --------- only extra
		1 \s leg \s bye
		1 \s wide ------- the other case also(runs)
'''
r = t.tr
extras = 0
noball = r"(\(no ball\) (FOUR|SIX|[0-3]))|((1) no ball)"
wide = r"(1) wide|\(wide\) (FOUR|SIX|[0-3])"
legbye = r"(FOUR|SIX|[0-3]) leg bye"
nb=0
wd=0
lb=0
for i in d["comments"].values():
	ma = re.search(noball+"|"+wide+"|"+legbye, i)
	print i,re.search(noball,i)
	# print nb
	if(re.search(noball,i)):
		nb+=1
	elif(re.search(wide,i)):
		wd+=1
	elif(re.search(legbye,i)):
		lb+=1
	# if(ma):
		# if(ma.group(1) == "FOUR"):
			# extras += 4
		# elif(ma.group(1) == "SIX"):
			# extras += 6
		# else:
			# extras +=int(ma.group(1))
print "(lb-"+str(lb)+", wd-"+str(wd)+", nb-"+str(nb)+")"
r.td("Extras")
r.td("(lb-"+str(lb)+", wd-"+str(wd)+", nb-"+str(nb)+")")
r.td(str(lb+wd+nb))

open("batting_stats.html","w").write(str(h))
'''
	Bowler Stats:
		o-m-r-w
'''

h = HTML()
t = h.table(border='1')
r=t.tr
r.th("Bowler")
r.th("Overs")
r.th("Maidens")
r.th("Runs")
r.th("Wickets")
for i in d["bowlers"].items():
	r = t.tr
	r.td(i[0])
	r.td(i[1].split("-")[0])
	r.td(i[1].split("-")[1])
	r.td(i[1].split("-")[2])
	r.td(i[1].split("-")[3])
open("bowling_stats.html","w").write(str(h))