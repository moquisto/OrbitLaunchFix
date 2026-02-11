import random
import math 
import matplotlib.pyplot as plt

xy0 = [0,0]
steps = 10
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

M = 10000
N = list(range(1,101))
R_sqrt = []                    # now: success fraction for each N
RMS_endtoend_distance = []     # now: standard error (SE) of the fraction

for totstep in range(len(N)):
    RforM = []                 # will hold 1 (success) or 0 (fail) for each trial

    for times in range(M):
        xyhistory = [xy0]
        visited = { (0,0) }    # SAW: remember visited sites
        success = True

        for steps in range(0, N[totstep]):
            next_pos = onestep(xyhistory[steps])
            tup = (next_pos[0], next_pos[1])
            if tup in visited:         # revisit → terminate & discard (unsuccessful)
                success = False
                break
            visited.add(tup)
            xyhistory.append(next_pos)

        RforM.append(1 if success else 0)

    mean_R2 = sum(RforM) / M            # here: fraction of successful walks p̂(N)

    summ = sum((value - mean_R2) ** 2 for value in RforM)
    var_R2 = summ / (M - 1)             # sample variance of Bernoulli( p̂ )
    se_R2 = math.sqrt(var_R2) / math.sqrt(M)   # SE of the fraction

    rms = mean_R2                        # keep variable names; rms now = fraction
    se_rms = se_R2                       # SE of the fraction

    R_sqrt.append(rms)
    RMS_endtoend_distance.append(se_rms)

for n_val, frac, se in zip(N, R_sqrt, RMS_endtoend_distance):
    print(f"N={n_val:4d}  success_fraction={frac:.5f}  SE={se:.5f}")

# Plot 1: fraction of successful SAWs vs N
plt.figure()
plt.plot(N, R_sqrt, 'o-')
plt.xscale('log')            # keep log-x; fraction is 0..1 so use linear y
plt.xlabel('N (steps)')
plt.ylabel('Success fraction')
plt.title('SAW success fraction vs N')
plt.tight_layout()
plt.show()

# Plot 2: standard error of the fraction vs N
plt.figure()
plt.plot(N, RMS_endtoend_distance, 'o-')
plt.xscale('log')
plt.xlabel('N (steps)')
plt.ylabel('Standard error of success fraction')
plt.title('SE of SAW success fraction vs N')
plt.tight_layout()
plt.show()