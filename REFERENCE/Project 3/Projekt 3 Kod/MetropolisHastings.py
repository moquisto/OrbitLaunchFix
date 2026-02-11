#implement the metropolis algorithm using the presentation steps

import random
import numpy

#do this for P(x) = xe^-x
# f(x) = x, meaning that the integral we are calculating is x^2e^-x on top and xe^-x on the bottom

def generateSamples(sampleSize, delta):
    #generate samples
    samples = []
    prev = 1 #initial guess
    i = 0
    acceptCounter = 0
    rejectCounter = 0
    while i < sampleSize:
        trial = prev + random.uniform(-delta, delta)
        w = (trial*(numpy.e ** (-trial))) / (prev*(numpy.e ** (-prev)))
        r = random.uniform(0,1)
        if trial <= 0:
            rejectCounter += 1
            samples.append(prev)
            continue
        if w > r:
            #accept trial
            prev = trial
            samples.append(trial)
            acceptCounter +=1
            i += 1
        else:
            rejectCounter += 1
        samples.append(prev)
        
    acceptanceRatio = acceptCounter / (acceptCounter + rejectCounter)
    return samples, acceptanceRatio

def metropolis(sampleSize = 10000, delta = 1):
    samples, acceptanceRatio = generateSamples(sampleSize, delta)
    standardDeviation = numpy.std(samples)
    N0 = 10 #skip 10 initial points
    sum = 0
    N = len(samples)
    standardError = standardDeviation / numpy.sqrt(N)
    while N0 < N:
        sum += samples[N0-1]
        N0 += 1
    res = sum / N
    print("res = " + str(res))
    print("acceptance ratio = " + str(acceptanceRatio))
    print("standard error = standard deviation / sqrt(N) = " + str(standardError))

if __name__ == "__main__":
    metropolis(100000, 10)


#I get the right answer from this. Nice. Couldn't you just use a normal montecarlo though?
#perhaps too inefficient