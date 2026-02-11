import random
import math 
import matplotlib.pyplot as plt

xy0 = [0,0]
steps = 10
xyhistory = [xy0]
last_pos = None   # <-- existing

def onestep(xymatrix):
    global last_pos
    # draw until we DON'T step back to the previous site
    while True:
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

        # if this proposal is the immediate back-step, redraw
        if last_pos is not None and xymatrix[0] == last_pos[0] and xymatrix[1] == last_pos[1]:
            continue
        break

    # after accepting the move, remember the site we came from
    last_pos = xyhistory[-1] if xyhistory else None
    return xymatrix

# --- added: plain random-walk step (no SAW and back-steps allowed) ---
def onestep_rw(xymatrix):
    val = random.randrange(4)
    xy = xymatrix.copy()
    if val == 0:
        xy[0] = xy[0] + 1
    elif val == 1:
        xy[0] = xy[0] - 1
    elif val == 2:
        xy[1] = xy[1] + 1
    else:
        xy[1] = xy[1] - 1
    return xy
# --------------------------------------------------------------------

M = 10000
N = list(range(1,20))
R_sqrt = []                    # success fraction for each N
RMS_endtoend_distance = []     # standard error (SE) of the fraction
RMS_SUCCESS = []    # SAW (no back-steps): sqrt(<R^2>) for successful walks
RMS_RW      = []    # Plain RW: sqrt(<R^2>)

for totstep in range(len(N)):
    RforM = []                 # 1 (success) or 0 (fail) for each trial
    R2_success_vals = []       # SAW R^2 for successful walks
    R2_rw_vals = []            # <-- added: RW R^2 for all walks

    for times in range(M):
        # --- SAW (no back-steps) ---
        xyhistory = [xy0]
        visited = { (0,0) }
        success = True
        last_pos = None        # reset per walk

        for steps in range(0, N[totstep]):
            next_pos = onestep(xyhistory[steps])
            tup = (next_pos[0], next_pos[1])
            if tup in visited:
                success = False
                break
            visited.add(tup)
            xyhistory.append(next_pos)

        RforM.append(1 if success else 0)
        if success:
            R2_success_vals.append(xyhistory[-1][0]**2 + xyhistory[-1][1]**2)

        # --- Plain random walk (baseline) ---
        xyhistory_rw = [xy0]
        for _ in range(N[totstep]):
            xyhistory_rw.append(onestep_rw(xyhistory_rw[-1]))
        R2_rw_vals.append(xyhistory_rw[-1][0]**2 + xyhistory_rw[-1][1]**2)

    # success fraction + SE (unchanged names)
    mean_R2 = sum(RforM) / M
    summ = sum((value - mean_R2) ** 2 for value in RforM)
    var_R2 = summ / (M - 1)
    se_R2 = math.sqrt(var_R2) / math.sqrt(M)
    rms = mean_R2
    se_rms = se_R2

    R_sqrt.append(rms)
    RMS_endtoend_distance.append(se_rms)

    RMS_SUCCESS.append(
        math.sqrt(sum(R2_success_vals)/len(R2_success_vals)) if R2_success_vals else float('nan')
    )
    RMS_RW.append(
        math.sqrt(sum(R2_rw_vals)/len(R2_rw_vals))
    )

for n_val, frac, se in zip(N, R_sqrt, RMS_endtoend_distance):
    print(f"N={n_val:4d}  success_fraction={frac:.5f}  SE={se:.5f}")

# Plot 1: fraction of successful SAWs vs N (unchanged)
plt.figure()
plt.plot(N, R_sqrt, 'o-')
plt.xscale('log')
plt.xlabel('N (steps)')
plt.ylabel('Success fraction')
plt.title('SAW success fraction vs N (no back-steps)')
plt.tight_layout()
plt.show()

# Plot 2: standard error of the fraction vs N (unchanged)
plt.figure()
plt.plot(N, RMS_endtoend_distance, 'o-')
plt.xscale('log')
plt.xlabel('N (steps)')
plt.ylabel('Standard error of success fraction')
plt.title('SE of SAW success fraction vs N')
plt.tight_layout()
plt.show()


plt.figure()
plt.plot(N, RMS_SUCCESS, 'o-', label='no back steps walk sqrt(<R^2>)')
plt.plot(N, RMS_RW,      's--', label='normal walk sqrt(<R^2>)')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('N')
plt.ylabel('RMS end-to-end distance sqrt(<R^2>)')
plt.title('RMS end-to-end distance vs N')
plt.legend()
plt.tight_layout()
plt.show()