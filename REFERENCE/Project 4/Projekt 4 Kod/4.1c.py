

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

L = 10

# The grid is n+1 points along x and y, including boundary points 0 and n
n = 10

# The grid spacing is L/n

# The number of iterations
nsteps = 56

# Initialize the grid to 0
v = np.zeros((n+1, n+1))
vnew = np.zeros((n+1, n+1))

# Set the boundary conditions
for i in range(1,n):
    v[0,i] = 10.0 # top
    v[n,i] = 10.0 # bottom
    v[i,0] = 5 # left
    v[i,n] = 5 # right

# Initialize the grid to 0.1 times 10 = 9
v[1:n, 1:n] = 7.5



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
# anim = animation.FuncAnimation(fig, update, frames=nsteps+1, interval=200, blit=True, repeat=False)
# plt.show()






# relaxation until 1% relative change
tol = 0.01       # 1% as a fraction
step = 0
v_old = v.copy()

while True:
    relax(n, v, checker=1)   # one relaxation iteration
    step += 1

    # absolute change between iterations interoir only 
    delta = np.abs(v[1:n, 1:n] - v_old[1:n, 1:n])
    num = np.max(delta)                      # max change
    den = np.max(np.abs(v_old[1:n, 1:n]))    # max magnitude of old values

    rel_error = num / den

    # prepare for next iteration
    v_old[:, :] = v

    if rel_error <= tol:
        break

print("number of iterations:", step)



# equipotential plot from converged v
x = np.linspace(0, L, n+1)
y = np.linspace(0, L, n+1)
X, Y = np.meshgrid(x, y)

plt.figure()
cf = plt.contourf(X, Y, v, levels=30)
plt.colorbar(cf, label='V (volts)')

cs = plt.contour(X, Y, v, levels=[5, 6, 7, 8, 9, 10],
                 colors='k', linewidths=0.5)
plt.clabel(cs, inline=True, fontsize=8)

plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('x')
plt.ylabel('y')
plt.title('equipotential lines, sides = 5, 10, 5, 10 V')

plt.show()
"""





import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

L = 10

# The grid is n+1 points along x and y, including boundary points 0 and n
n = 10

# The grid spacing is L/n

# The number of iterations
nsteps = 100

# Initialize the grid to 0
v = np.zeros((n+1, n+1))
vnew = np.zeros((n+1, n+1))

# Set the boundary conditions
for i in range(1,n):
    v[0,i] = 10.0 # top
    v[n,i] = 0 # bottom
    v[i,0] = 10 # left
    v[i,n] = 10 # right

# Initialize the grid to 0.1 times 10 = 9
v[1:n, 1:n] = 7



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
# anim = animation.FuncAnimation(fig, update, frames=nsteps+1, interval=200, blit=True, repeat=False)
# plt.show()






# relaxation until 1% relative change
tol = 0.0001       # 1% as a fraction
step = 0
v_old = v.copy()

while True:
    relax(n, v, checker=1)   # one relaxation iteration
    step += 1

    # absolute change between iterations interoir only 
    delta = np.abs(v[1:n, 1:n] - v_old[1:n, 1:n])
    num = np.max(delta)                      # max change
    den = np.max(np.abs(v_old[1:n, 1:n]))    # max magnitude of old values

    rel_error = num / den

    # prepare for next iteration
    v_old[:, :] = v

    if rel_error <= tol:
        break

print("number of iterations:", step)



# equipotential plot from converged v
x = np.linspace(0, L, n+1)
y = np.linspace(0, L, n+1)
X, Y = np.meshgrid(x, y)

plt.figure()
cf = plt.contourf(X, Y, v, levels=30)
plt.colorbar(cf, label='V (volts)')

cs = plt.contour(X, Y, v, levels=[5, 6, 7, 8, 9, 10],
                 colors='k', linewidths=0.5)
plt.clabel(cs, inline=True, fontsize=8)

plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('x')
plt.ylabel('y')
plt.title('equipotential lines, sides = 5, 10, 5, 10 V')

plt.show()

"""