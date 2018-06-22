#CSV Reaser
	csvreader = csv.reader(open("data/Advertising.csv"))
	
	x = []
	y = []
	next(csvreader)
	
	for line in csvreader:
	    x_i = [float(line[1]),float(line[2]),float(line[3])]
	    x.append(x_i)
	    y_i = float(line[4])
	    y.append(y_i)
	
	X = np.array(x)
	Y = np.array(y)

#txt file reading and 
def read_data():
    words = []
    with open("words.txt", "r") as filestream:
        for line in filestream:
            currentline = line.split(",",)
            word = [currentline[0],int(currentline[1].rstrip())] #rstrip() : delete. \n
            print(word)
            words.append(word)
    # [[단어1, 빈도수], [단어2, 빈도수] ... ]형으로 변환해 리턴합니다.
    return words