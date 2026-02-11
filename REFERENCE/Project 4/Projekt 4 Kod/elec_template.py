import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

L = 25.0

# The grid is n+1 points along x and y, including boundary points 0 and n
n = 25

# The grid spacing is L/n

# The number of iterations
nsteps = 100

# Initialize the grid to 0
v = np.zeros((n+1, n+1))
vnew = np.zeros((n+1, n+1))

# Set the boundary conditions
for i in range(1,n):
    v[0,i] = 10.0
    v[n,i] = 10.0
    v[i,0] = 10.0
    v[i,n] = 10.0

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(v, cmap=None, interpolation='nearest')
fig.colorbar(im)

# checker=1: no checkboard, checker=2: checkerboard (note: n should be even)
checker = 1

# perform one step of relaxation
def relax(n, v, checker):
    for check in range(0,checker):
        for x in range(1,n):
            for y in range(1,n):
                if (x*(n+1) + y) % checker == check:
                    vnew[x,y] = (v[x-1][y] + v[x+1][y] + v[x][y-1] + v[x][y+1])*0.25

        # Copy back the new values to v
        # Note that you can directly store in v instead of vnew with Gauss-Seidel or checkerboard
        for x in range(1,n):
            for y in range(1,n):
                if (x*(n+1) + y) % checker == check:
                    v[x,y] = vnew[x,y]

def update(step):
    print(step)
    global n, v, checker

    # FuncAnimation calls update several times with step=0,
    # so we needs to skip the update with step=0 to get
    # the correct number of steps 
    if step > 0:
        relax(n, v, checker)

    im.set_array(v)
    return im,

# we generate nsteps+1 frames, because frame=0 is skipped (see above)
anim = animation.FuncAnimation(fig, update, frames=nsteps+1, interval=200, blit=True, repeat=False)
plt.show()

