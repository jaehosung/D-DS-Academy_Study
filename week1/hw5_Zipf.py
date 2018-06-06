from scipy.stats import linregress
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import elice_utils
import math
import operator

def main():    
    data = read_data()
    data = sorted(data,key=lambda data: data[1],reverse = True)
    for i in range(len(data)):
        #Rank start from 1
        data[i][0] = math.log(i+1)
        data[i][1] = math.log(data[i][1])
    data = np.array(data)
    slope,intercept = do_linear_regression(data[:,0],data[:,1])
    
    return slope,intercept
def read_data():
    words=[]
    with open('words.txt') as fp:
        for line in fp:
            line_data = line.strip().split(',')
			# make the string data to float data in a line
            line_data_numeric = [line_data[0],float(line_data[1])]
            words.append(line_data_numeric)
    return words

def draw_chart(X, Y, slope, intercept):
    fig = plt.figure()
    
    # 여기에 내용을 채우세요.
    
    plt.savefig('chart.png')
    elice_utils.send_image('chart.png')

def do_linear_regression(X, Y):
    # 여기에 내용을 채우세요.
    slope,intercept,r_value,p_value,std_err = linregress(X,Y)

    return slope, intercept

if __name__ == "__main__":
    main()
