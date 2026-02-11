import random

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

for n in range(0,steps):
    xyhistory.append(onestep(xyhistory[n]))

print(xyhistory)



import matplotlib.pyplot as plt

def run_walk(N):
    xy = [0, 0]
    hist = [xy]
    for n in range(N):
        hist.append(onestep(hist[n]))
    return hist

for N in [10, 100, 1000]:
    H = run_walk(N)
    xs = [p[0] for p in H]
    ys = [p[1] for p in H]

    plt.figure()
    plt.plot(xs, ys, '-o', ms=2)
    plt.scatter([0], [0], s=30, c='k', label='start')
    plt.scatter([xs[-1]], [ys[-1]], s=30, c='r', label='end')
    plt.axis('equal')
    plt.xlabel('x'); plt.ylabel('y')
    plt.title(f'random walk, N={N}')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()