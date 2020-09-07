f1 = open("test.txt","r",encoding="utf-8-sig").readlines()
f2 = open("test1.txt","w",encoding="utf-8-sig")
for line in f1:
	if(len(line)<=40):
		str = "0\t"+line
	if(len(line)>40):
		str = "0\t"+line[-40:]
	f2.write(str)
