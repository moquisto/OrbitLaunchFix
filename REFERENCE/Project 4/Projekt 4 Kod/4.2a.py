import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

L = 10

# The grid is n+1 points along x and y, including boundary points 0 and n
n = 20

# The grid spacing is L/n

# The number of iterations
nsteps = 1000

# Initialize the grid to 0
v = np.zeros((n+1, n+1))
vnew = np.zeros((n+1, n+1))

# Set the boundary conditions
for i in range(1,n):
    v[0,i] = 10.0
    v[n,i] = 10.0
    v[i,0] = 10.0
    v[i,n] = 10.0

    # Initialize the grid to 0.1 times 10 = 9

    v[1:n, 1:n] = 9.0

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(v, cmap=None, interpolation='nearest')
fig.colorbar(im)

# checker=1: no checkboard, checker=2: checkerboard (note: n should be even)
checker = 1

# perform one step of sequential
def relax(n, v):
    for x in range(1, n):
        for y in range(1, n):
            # directly overwrite v[x,y] with the average of its neighbours
            v[x, y] = 0.25 * (
                v[x-1, y] +  # left   (alread updated this sweep when x > 1)
                v[x+1, y] +  # right  (old value when x < n-1)
                v[x, y-1] +  # below  (already updated when y > 1)
                v[x, y+1]    # above  (old value when y < n-1)
            )


def update(step):
    print(step)
    global n, v

    # FuncAnimation calls update several times with step=0,
    # so we needs to skip the update with step=0 to get
    # the correct number of steps 
    if step > 0:
        relax(n, v)

    im.set_array(v)
    return im,

# we generate nsteps+1 frames, because frame=0 is skipped (see above)
# anim = animation.FuncAnimation(fig, update, frames=nsteps+1, interval=200, blit=True, repeat=False)
# plt.show()


target = 10.0
tol = 0.01 * target   # 0.1 V
step = 0

while True:
    relax(n, v)              # one gauss–seidel 
    step += 1

    # compute max error in interior
    error = np.max(np.abs(v[1:n, 1:n] - target))

    if error <= tol:
        break

print("number of iterations:", step)

