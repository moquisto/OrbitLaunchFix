import random
import numpy as np
import matplotlib.pyplot as plt

def generateSamples(sampleSize, delta):
    samples = []
    prev = 1.0
    accept = 0
    reject = 0

    while len(samples) < sampleSize:
        trial = prev + random.uniform(-delta, delta)

        if trial <= 0:
            samples.append(prev)
            reject += 1
            continue

        w = (trial * np.exp(-trial)) / (prev * np.exp(-prev))
        r = random.uniform(0, 1)

        if r < w:
            prev = trial
            accept += 1
        else:
            reject += 1
        
        samples.append(prev)

    acceptanceRatio = accept / (accept + reject)
    return samples, acceptanceRatio


def metropolis(sampleSize=10000, delta=1):
    samples, acceptanceRatio = generateSamples(sampleSize, delta)

    burn = 100
    samples = samples[burn:]

    N = len(samples)
    mean = sum(samples) / N
    std = np.std(samples) #standard deviation
    stderr = std / np.sqrt(N)

    #print("Mean =", mean)
    #print("Acceptance ratio =", acceptanceRatio)
    #print("Std. error =", stderr)
    return stderr, mean, acceptanceRatio


if __name__ == "__main__":
    deltaList = []
    errorList = []
    RMSDiffList = []
    acceptanceRatioList = []
    deltaVal = 0.1
    while deltaVal <= 10:
        errSum = 0
        RMSDiffSum = 0 #sum RMSs
        acceptanceRatioSum = 0
        for i in range(15):
            stdError , mean, acceptanceR = metropolis(10000, deltaVal)
            errSum += stdError
            RMSDiffSum += np.sqrt((2 - mean)**2) #add RMS of single run
            acceptanceRatioSum += acceptanceR

        deltaList.append(deltaVal)
        errorList.append(errSum / 15)
        RMSDiffList.append((RMSDiffSum / 15)) #add average RMS
        acceptanceRatioList.append(acceptanceRatioSum/15)

        deltaVal += 0.1
    #compare standard error with root mean square of the difference bw 2 and computed answer
    plt.clf()
    #plt.plot(deltaList, errorList, label = "Standard error")
    #plt.plot(deltaList, RMSDiffList, label = "Average RMS difference")
    plt.plot(deltaList, acceptanceRatioList, label = "Acceptance ratio")
    plt.legend()
    #plt.title("Standard error and RMS vs delta")
    plt.title("Average Acceptance Ratio vs delta")
    plt.show()
    

