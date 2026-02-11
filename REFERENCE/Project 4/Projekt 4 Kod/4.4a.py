import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

L = 10

# The grid is n+1 points along x and y, including boundary points 0 and n
n = 10

# The grid spacing is L/n

# The number of iterations (used only if you still want to animate)
nsteps = 100

# Initialize the grid to 0
v = np.zeros((n+1, n+1))
vnew = np.zeros((n+1, n+1))

# Set the boundary conditions (same geometry as 4.3)
for i in range(1, n):
    v[0, i] = 10.0  # top
    v[n, i] = 10.0  # bottom
    v[i, 0] = 5     # left
    v[i, n] = 5     # right

# Initial guess in the interior (doesn't matter for G, but we keep structure)
v[1:n, 1:n] = 7.5

"""
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(v, cmap=None, interpolation='nearest')
fig.colorbar(im)
"""

# checker=1: no checkboard, checker=2: checkerboard (note: n should be even)
checker = 1

# perform one step of relaxation (Jacobi style)
def relax(n, v, checker):
    for check in range(0, checker):
        for x in range(1, n):
            for y in range(1, n):
                if (x*(n+1) + y) % checker == check:
                    vnew[x, y] = 0.25 * (
                        v[x-1, y] + v[x+1, y] + v[x, y-1] + v[x, y+1]
                    )

        # Copy back the new values to v
        for x in range(1, n):
            for y in range(1, n):
                if (x*(n+1) + y) % checker == check:
                    v[x, y] = vnew[x, y]

"""
def update(step):
    print(step)
    global n, v, checker

    if step > 0:
        relax(n, v, checker)

    im.set_array(v)
    return im,
"""

# we generate nsteps+1 frames, because frame=0 is skipped (see above)
# anim = animation.FuncAnimation(fig, update, frames=nsteps+1, interval=200, blit=True, repeat=False)
# plt.show()

# Relaxation until 1% relative change 
tol = 0.01       # 1% as a fraction
step = 0
v_old = v.copy()

while True:
    relax(n, v, checker=1)   # one relaxation iteration
    step += 1

    # absolute change between iterations (interior only)
    delta = np.abs(v[1:n, 1:n] - v_old[1:n, 1:n])
    num = np.max(delta)                      # max change
    den = np.max(np.abs(v_old[1:n, 1:n]))    # max magnitude of old values

    rel_error = num / den

    # prepare for next iteration
    v_old[:, :] = v

    if rel_error <= tol:
        break

# print("number of relaxation iterations:", step)


# greens function 

# build a list of all boundary sites 
boundary_sites = []

# top edge (0, j)
for j in range(0, n+1):
    boundary_sites.append((0, j))

# bottom edge (n, j)
for j in range(0, n+1):
    boundary_sites.append((n, j))

# left edge (i, 0)
for i in range(1, n):
    boundary_sites.append((i, 0))

# right edge (i, n)
for i in range(1, n):
    boundary_sites.append((i, n))

nb = len(boundary_sites)   # number of distinct boundary points
print("number of boundary sites:", nb)

# map (i,j) to boundary index b
b_index = { (i, j): b for b, (i, j) in enumerate(boundary_sites) }

# one random walk
def exit_boundary_index(i0, j0, n):
    """
    One random walk starting at interior point (i0, j0).
    Returns the index b of the boundary site where it first exits.
    """
    i, j = i0, j0

    # walk until we hit any boundary
    while 0 < i < n and 0 < j < n:
        r = np.random.randint(4)
        if   r == 0: i += 1   # down
        elif r == 1: i -= 1   # up
        elif r == 2: j += 1   # right
        else:        j -= 1   # left

    return b_index[(i, j)]   # look up which boundary site this is

# build G(i,j; x_b, y_b) for all interior sites and all boundary sites
nwalkers = 200                          
G = np.zeros((n+1, n+1, nb))              # G[i,j,b]

for i0 in range(1, n):                    # interior rows
    for j0 in range(1, n):                # interior columns
        counts = np.zeros(nb, dtype=int)
        for k in range(nwalkers):
            b = exit_boundary_index(i0, j0, n)
            counts[b] += 1
        # normalise to probabilities
        G[i0, j0, :] = counts / nwalkers


# save G and information about the boundary sites
G_flat = G.reshape(( (n+1)*(n+1), nb ))
np.savetxt("G_matrix.txt", G_flat, fmt="%.6f")

# save boundary coordinates (i_b, j_b) for each column b
boundary_array = np.array(boundary_sites, dtype=int)
np.savetxt("G_boundary_sites.txt", boundary_array, fmt="%d")

