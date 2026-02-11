import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

L = 10

# The grid is n+1 points along x and y, including boundary points 0 and n
n = 10

# The grid spacing is L/n


nboundary = 40      # should match file

# load freens from 4.4a 
G_flat = np.loadtxt("G_matrix.txt")
boundary_sites = np.loadtxt("G_boundary_sites.txt", dtype=int)

# reshape back to G[i,j,b]
G = G_flat.reshape((n+1, n+1, nboundary))

print("loaded G with shape:", G.shape)
print("number of boundary sites:", boundary_sites.shape[0])

# which index is which site
print("\nboundary index = (i_b, j_b):")
for b, (ib, jb) in enumerate(boundary_sites):
    print(f"b = {b:2d}  -> (i_b, j_b) = ({ib}, {jb})")


# build baseline boundary potentials 5/10/5/10
def build_baseline_Vb():
    Vb = np.zeros(nboundary)
    for b, (ib, jb) in enumerate(boundary_sites):
        if ib == 0 or ib == n:   # top/bottom
            Vb[b] = 10.0
        else:                    # left/right
            Vb[b] = 5.0
    return Vb


# compute interior potential for a chosen 20V set of sites
def potential_for_hot_sites(hot_indices):
    """
    chosn 20V sites are boundary indices b where V_b = 20 V
    all other boundary sites use the baseline 5/10/5/10.
    Returns V grid (n+1 x n+1).
    """
    Vb = build_baseline_Vb()
    for b in hot_indices:
        Vb[b] = 20.0

    # V(i,j) = sum_b G(i,j,b) * V_b
    V_total = np.zeros((n+1, n+1))
    for b in range(nboundary):
        V_total += G[:, :, b] * Vb[b]

    # set boundary explicitly so we dont have 0 on the boundary and instead boundary vals
    for b, (ib, jb) in enumerate(boundary_sites):
        V_total[ib, jb] = Vb[b]

    return V_total


# POINT 1: 5 sites for point (3,5)

target_1 = (3, 5)

# chosen 20V sites 
hot_35 = [4, 5, 6, 24, 33]

V35 = potential_for_hot_sites(hot_35)
print(f"\nfor target point (3,5) and chosen sites {hot_35}:")
print(f"  V(3,5) = {V35[target_1]:.4f} V")

# plot
x = np.linspace(0, L, n+1)
y = np.linspace(0, L, n+1)
X, Y = np.meshgrid(x, y)

plt.figure()
cf = plt.contourf(X, Y, V35, levels=30)
plt.colorbar(cf, label="V")
plt.title("potential with chosen 5 20V boundary sites (target (3,5))")
plt.gca().set_aspect("equal", adjustable="box")

# mark the target and the hot boundary sites
plt.scatter([target_1[1] * L / n], [target_1[0] * L / n],
            c="red", s=40, edgecolors="black", label="target (3,5)")

for b in hot_35:
    ib, jb = boundary_sites[b]
    plt.scatter([jb * L / n], [ib * L / n],
                c="white", s=40, edgecolors="black")

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()




# POINT 1: 5 sites for point (5,3)

target_2 = (5, 3)

# chosen 20V sites 
hot_53 = [24, 25, 26, 27, 28] 

V53 = potential_for_hot_sites(hot_53)
print(f"\nfor target point (5,3) and chosen sites {hot_35}:")
print(f"  V(5,3) = {V53[target_2]:.4f} V")

plt.figure()
cf = plt.contourf(X, Y, V53, levels=30)
plt.colorbar(cf, label="V ")
plt.title("potential with chosen 5 20V boundary sites (target (5, 3))")
plt.gca().set_aspect("equal", adjustable="box")

plt.scatter([target_2[1] * L / n], [target_2[0] * L / n],
            c="red", s=40, edgecolors="black", label="target (5,3)")

for b in hot_53:
    ib, jb = boundary_sites[b]
    plt.scatter([jb * L / n], [ib * L / n],
                c="white", s=40, edgecolors="black")

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()