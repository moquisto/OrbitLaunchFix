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
    v[n,i] = 10.0 # bottom
    v[i,0] = 5 # left
    v[i,n] = 5 # right

# Initialize the grid to 0.1 times 10 = 9
v[1:n, 1:n] = 7.5

"""
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(v, cmap=None, interpolation='nearest')
fig.colorbar(im)
"""

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

"""
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
"""

# we generate nsteps+1 frames, because frame=0 is skipped (see above)
# anim = animation.FuncAnimation(fig, update, frames=nsteps+1, interval=200, blit=True, repeat=False)
# plt.show()


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




def random_walk_potential(i0, j0, n, nwalkers):
    hits = []
    for k in range(nwalkers):
        i, j = i0, j0
        while 0 < i < n and 0 < j < n:
            r = np.random.randint(4)
            if   r == 0: i += 1   # down
            elif r == 1: i -= 1   # up
            elif r == 2: j += 1   # right
            else:        j -= 1   # left

        if i == 0 or i == n:      # top or bottom
            Vb = 10.0
        else:                     # left or right
            Vb = 5.0
        hits.append(Vb)

    return np.mean(hits)



def averaged_random_walk(i0, j0, n, nwalkers, n_runs):
    vals = []
    for r in range(n_runs):
        vals.append(random_walk_potential(i0, j0, n, nwalkers))
    vals = np.array(vals)
    return vals.mean(), vals.std()




# interior points chosen:
points = [
    ("centre",          n // 2,     n // 2),   # middle of the square

    ("midway_x_left",   n // 2,     n // 4),   # halfway between centre and left
    ("near_left",       n // 2,     1),        # very close to left wall x

    ("midway_y_top",    n // 4,     n // 2),   # halfway between centre and top
    ("near_top",        1,          n // 2),   # very close to top wall y
]

# number of walkers to test – you can add more if you like
N_values = (100, 1000)

for label, i0, j0 in points:
    V_relax = v[i0, j0]
    print(f"\nPoint {label} at (i={i0}, j={j0})")
    print(f"   V_relax = {V_relax:.4f}")

    for Nwalk in N_values:
        mean_rw, std_rw = averaged_random_walk(i0, j0, n, Nwalk, n_runs=100)
        print(
            f"   Nwalk = {Nwalk:4d}: "
            f"V_randomwalk = {mean_rw:.4f} ± {std_rw:.4f},  "
            f"mean difference = {mean_rw - V_relax:.4f}"
        )
    





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

# mark choosen points 
xs = []
ys = []
labels_plot = []

for label, i0, j0 in points:
    # convert grid indices (i,j) to physical coordinates (x,y)
    x_coord = j0 * L / n   # column to x
    y_coord = i0 * L / n   # row to y
    xs.append(x_coord)
    ys.append(y_coord)
    labels_plot.append(label)

# plot the points
plt.scatter(xs, ys, c="red", s=40, edgecolors="black")

# add small text labels next to each point
for x_coord, y_coord, label in zip(xs, ys, labels_plot):
    plt.text(x_coord + 0.2, y_coord + 0.2, label,
             color="white", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))
# ----------------------------------------

plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('x')
plt.ylabel('y')
plt.title('equipotential lines, sides = 5, 10, 5, 10 V')

plt.show()