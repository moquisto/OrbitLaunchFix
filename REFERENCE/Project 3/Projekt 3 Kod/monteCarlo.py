#write a normal uniform montecarlo - simple algorithm

import math
import numpy
import matplotlib
import random


def monteCarlo(start = 0, end = 1, iter = 100000):
    #do it for xe^x
    sum = 0
    for i in range(iter):
        x = random.uniform(start, end)
        sum += x * (numpy.e ** x)
    res = sum / iter
    print(res)
#works decently

if __name__ == "__main__":
    monteCarlo()