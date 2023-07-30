import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Takes CSV file as input with columns of x and y as the variables for the data.
# Upon creation of spline object, everything is automatically solved. Can then use self.function() to obtain the coefficients of function. Coefficients are arranged in a matrix by (x^3, x^2, x, constant) per data interval, ascending.


class spline:

    def __init__(self):
        self.coeff = []
        self.points = []
        data = pd.read_csv("data")
        data['x'].tolist().sort()
        data['y'].tolist().sort()
    
    def matchleft(self, x, y):
        for i in range(len(y) - 1): #Gp through n-1 points
            self.points.append(y[i]) #b
            temp = [x[i]**3, x[i]**2, x[i], 1] #each row of A
            for j in range(len(y) - i - 2): #add the trailing zeroes
                temp += [0, 0, 0, 0]
            for k in range(i): #add the leading zeroes
                temp = [0, 0 ,0 ,0] + temp
            self.coeff.append(temp) #adding rows of A

    def matchright(self, x, y):
        for i in range(1, len(y)): #Gp through n-1 points
            self.points.append(y[i]) #b
            temp = [x[i]**3, x[i]**2, x[i], 1] #each row of A
            for j in range(len(y) - i - 1): #add the trailing zeroes
                temp += [0, 0, 0, 0]
            for k in range(i-1): #add the leading zeroes
                temp = [0, 0 ,0 ,0] + temp
            self.coeff.append(temp) #adding rows of A
    
    def matchderiv(self, x, y):
        for i in range(1, len(y)-1): #Gp through n-1 points
            self.points.append(0) #b
            temp = [3*x[i]**2, 2*x[i], 1, 0, -3*x[i]**2, -2*x[i], -1, 0] #each row of A
            for j in range(len(y) - i - 2): #add the trailing zeroes
                temp += [0, 0, 0, 0]
            for k in range(i-1): #add the leading zeroes
                temp = [0, 0 ,0 ,0] + temp
            self.coeff.append(temp) #adding rows of A

    def match2deriv(self, x, y):
        for i in range(1, len(y)-1): #Go through n-1 points
            self.points.append(0) #b
            temp = [6*x[i], 2, 0, 0, -6*x[i], -2, 0, 0] #each row of A
            for j in range(len(y) - i - 2): #add the trailing zeroes
                temp += [0, 0, 0, 0]
            for k in range(i-1): #add the leading zeroes
                temp = [0, 0 ,0 ,0] + temp
            self.coeff.append(temp) #adding rows of A

    def matchendpoints(self, x, y):
        self.points.append(0) #adding zeroes to b
        self.points.append(0)
        temp = [6*x[0], 2, 0, 0] #start point
        for i in range(len(y)-2): #add the trailing zeroes
            temp += [0, 0, 0, 0]
        self.coeff.append(temp)
        temp = [6*x[len(y)-1], 2, 0, 0] #end point
        for k in range(len(y)-2): #add the leading zeroes
            temp = [0, 0 ,0 ,0] + temp
        self.coeff.append(temp)
    
    def function(self):
        x = self.x
        y = self.y
        self.matchleft(x, y) #Functions to make proper matrix
        self.matchright(x, y)
        self.matchderiv(x, y)
        self.match2deriv(x, y)
        self.matchendpoints(x, y)
        b = np.array(self.points) #Making of augmented matrix
        b = b[:, np.newaxis]
        A = np.array(self.coeff)
        ans = (np.dot(np.linalg.inv(A), b)) #Solving for coefficients
        return ans
    
    def solve(self, ans, z):
        x = self.x
        if x[len(x)-1] == z: #Edge case when data point is at the end of interval
                a = len(x)-2
        else:
            for i in range(len(x)-1): #Find correct coefficients to use
                if x[i] <= z < x[i+1]:
                    a = i
                    break
        return (ans[a*4]*(z**3)) + (ans[a*4 + 1]*(z**2)) + (z*ans[a*4 + 2]) + (ans[a*4 + 3]) #Multiply coressponding coefficients