import random
import math 
import matplotlib.pyplot as plt

xy0 = [0,0]
xyhistory = [xy0]

def onestep(xymatrix):
    val = random.randrange(4)
    xymatrix = xymatrix.copy()
    if val == 0:
        xymatrix[0] = xymatrix[0] + 1
    elif val == 1:
        xymatrix[0] = xymatrix[0] - 1
    elif val == 2:
        xymatrix[1] = xymatrix[1] + 1
    else:
        xymatrix[1] = xymatrix[1] - 1
    return xymatrix

M = 1000
N = [1, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 
     600, 700, 800, 900, 1000]
R_sqrt = []
RMS_endtoend_distance = [] 

for totstep in range(len(N)):
    RforM = []

    for times in range(M):
        xyhistory = [xy0]
        for steps in range(0, N[totstep]):
            xyhistory.append(onestep(xyhistory[steps]))
        RforM.append(xyhistory[-1][0]**2 + xyhistory[-1][1]**2)

    mean_R2 = sum(RforM) / M


    summ = sum((value - mean_R2) ** 2 for value in RforM)
    var_R2 = summ / (M - 1)
    se_R2 = math.sqrt(var_R2) / math.sqrt(M)

    rms = math.sqrt(mean_R2)
    se_rms = se_R2 / (2.0 * rms)


    R_sqrt.append(rms)
    RMS_endtoend_distance.append(se_rms)

for n_val, rms, se in zip(N, R_sqrt, RMS_endtoend_distance):
    print(f"N={n_val:4d}  RMS=sqrt(<R^2>)={rms:.5f}  SE(RMS)={se:.5f}")

# print(xyhistory)

# Plot 1: RMS end-to-end distance vs N
plt.figure()
plt.plot(N, R_sqrt, 'o-')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('N')
plt.ylabel('RMS end-to-end distance  =  sqrt(<R^2>)')
plt.title('RMS vs N')
plt.tight_layout()
plt.show()

# Plot 2: Standard error estimate vs N
plt.figure()
plt.plot(N, RMS_endtoend_distance, 'o-')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('N')
plt.ylabel('Standard error of RMS')
plt.title('SE(RMS) vs N')
plt.tight_layout()
plt.show()