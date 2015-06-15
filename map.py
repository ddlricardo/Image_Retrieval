ap = []
for i in range(2000):
	orderdic = {}
	for j in range(100000):
		orderdic[j] = prediction[i][j]
	answer = sorted(orderdic.iteritems(), key=lambda d:d[1], reverse=True)
	apsum = float(0)
	rightsum = 0
	for j in range(100000):
		if groundtruth[i][answer[j][0]] == 1:
			rightsum += 1
			apsum += rightsum * 1.0 / (j + 1)
	ap.append(apsum / sum(groundtruth[i]))
print 'MAP = ' + str(sum(ap) / len(ap))