#!/bin/python3

# Template for traffic simulation
# BH, MP 2021-11-15, latest version 2024-11-08.

"""
    This template is used as backbone for the traffic simulations.
    Its structure resembles the one of the pendulum project, that is you have:
    (a) a class containing the state of the system and it's parameters
    (b) a class storing the observables that you want then to plot
    (c) a class that propagates the state in time (which in this case is discrete), and
    (d) a class that encapsulates the aforementioned ones and performs the actual simulation
    You are asked to implement the propagation rule(s) corresponding to the traffic model(s) of the project.
"""

import math
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy.random as rng
import numpy as np

import matplotlib


class Cars:

    """ Class for the state of a number of cars """

    def __init__(self, numCars=5, roadLength=50, v0=1):
        self.numCars    = numCars
        self.roadLength = roadLength
        self.t  = 0
        self.x  = []
        self.v  = []
        self.c  = []
        for i in range(numCars):
            # TODO: Set the initial position for each car.
            # Note that the ordering of the cars on the road needs to match
            # the order in which you compute the distances between cars

            pos = (i * roadLength) // numCars

            self.x.append(pos)        # the position of the cars on the road
            self.v.append(v0)       # the speed of the cars
            self.c.append(i)        # the color of the cars (for drawing)

    # NOTE: you can, but don't have to use this function for computing distances
    def distance(self, i):
        # TODO: Implement the function returning the PERIODIC distance 
        # between car i and the one in front 

        j = (i + 1) % self.numCars
        # subtract 1 so the leader's occupied cell is not counted as free space
        return (self.x[j] - self.x[i] - 1) % self.roadLength


class Observables:

    """ Class for storing observables """

    def __init__(self):
        self.time = []          # list to store time
        self.flowrate = []      # list to store the flow rate
        

class BasePropagator:

    def __init__(self):
        return
        
    def propagate(self, cars, obs):

        """ Perform a single integration step """
        
        fr = self.timestep(cars, obs)

        # Append observables to their lists
        obs.time.append(cars.t)
        obs.flowrate.append(fr)
              
    def timestep(self, cars, obs):

        """ Virtual method: implemented by the child classes """
        
        pass
      
        
class ConstantPropagator(BasePropagator) :
    
    """ 
        Cars do not interact: each position is just 
        updated using the corresponding velocity 
    """
    
    def timestep(self, cars, obs):
        for i in range(cars.numCars):
            cars.x[i] += cars.v[i]
        cars.t += 1
        return 0

class MyPropagator(BasePropagator):

    def __init__(self, vmax, p):
        BasePropagator.__init__(self)
        self.vmax = int(vmax)
        self.p = float(p)

    def timestep(self, cars, obs):
        N = cars.numCars
        L = cars.roadLength

        # Ensure consistent "car i -> car in front" ordering
        order = sorted(range(N), key=lambda i: cars.x[i])
        cars.x = [cars.x[i] for i in order]
        cars.v = [cars.v[i] for i in order]
        cars.c = [cars.c[i] for i in order]

        # 1) Accelerate toward vmax (synchronous: use temp list)
        v_new = [min(v + 1, self.vmax) for v in cars.v]

        # 2) Brake to avoid collisions: limit by periodic headway d_i
        for i in range(N):
            d_i = cars.distance(i)  # number of empty cells to the leader
            if v_new[i] > d_i:
                v_new[i] = d_i

        # 3) Random slow-down with prob p (only if moving)
        for i in range(N):
            if v_new[i] > 0 and rng.random() < self.p:
                v_new[i] -= 1

        # 4) Move all cars simultaneously on the ring
        x_new = [(cars.x[i] + v_new[i]) % L for i in range(N)]

        # Keep arrays ordered by position for the next step
        order2 = sorted(range(N), key=lambda i: x_new[i])
        cars.x = [x_new[i] for i in order2]
        cars.v = [v_new[i] for i in order2]
        cars.c = [cars.c[i] for i in order2]

        # Advance discrete time and return instantaneous flow rate
        cars.t += 1
        flow = float(sum(cars.v)) / L
        return flow

############################################################################################

def draw_cars(cars, cars_drawing):

    """ Used later on to generate the animation """
    theta = []
    r     = []

    for position in cars.x:
        # Convert to radians for plotting  only (do not use radians for the simulation!)
        theta.append(position * 2 * math.pi / cars.roadLength)
        r.append(1)

    return cars_drawing.scatter(theta, r, c=cars.c, cmap='hsv')


def animate(framenr, cars, obs, propagator, road_drawing, stepsperframe):

    """ Animation function which integrates a few steps and return a drawing """

    for it in range(stepsperframe):
        propagator.propagate(cars, obs)

    return draw_cars(cars, road_drawing),


class Simulation:

    def reset(self, cars=Cars()) :
        self.cars = cars
        self.obs = Observables()

    def __init__(self, cars=Cars()) :
        self.reset(cars)

    def plot_observables(self, title="simulation"):
        plt.clf()
        plt.title(title)
        plt.plot(self.obs.time, self.obs.flowrate)
        plt.xlabel('time')
        plt.ylabel('flow rate')
        plt.savefig(title + ".pdf")
        plt.show()

    # Run without displaying any animation (fast)
    def run(self,
            propagator,
            numsteps=200,           # final time
            title="simulation",     # Name of output file and title shown at the top
            ):

        for it in range(numsteps):
            propagator.propagate(self.cars, self.obs)

        self.plot_observables(title)

    # Run while displaying the animation of bunch of cars going in circe (slow-ish)
    def run_animate(self,
            propagator,
            numsteps=200,           # Final time
            stepsperframe=1,        # How many integration steps between visualising frames
            title="simulation",     # Name of output file and title shown at the top
            ):

        numframes = int(numsteps / stepsperframe)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.axis('off')
        # Call the animator, blit=False means re-draw everything
        anim = animation.FuncAnimation(plt.gcf(), animate,  # init_func=init,
                                       fargs=[self.cars,self.obs,propagator,ax,stepsperframe],
                                       frames=numframes, interval=50, blit=True, repeat=False)
        plt.show()

        # If you experience problems visualizing the animation and/or
        # the following figures comment out the next line 
        # plt.waitforbuttonpress(30)

        self.plot_observables(title)
    






# ====== 2.2(c) helpers: finite-size scan at constant density ======

def avg_flow(N, L, vmax=2, p=0.5, burn_in=500, measure=500, init_mode='even'):
    """Time-averaged flow for one (N,L) after burn-in. Burn-in scaled with L."""
    rng.seed(None)
    cars = Cars(numCars=N, roadLength=L, v0=0)
    if init_mode == 'jam':
        cars.x = list(range(cars.numCars)); cars.v = [0]*cars.numCars
    elif init_mode == 'random':
        pos = sorted(rng.choice(cars.roadLength, size=cars.numCars, replace=False).tolist())
        cars.x = pos; cars.v = [0]*cars.numCars
    sim  = Simulation(cars)
    prop = MyPropagator(vmax=vmax, p=p)

    for _ in range(burn_in):
        prop.propagate(sim.cars, sim.obs)
    sim.obs.time.clear(); sim.obs.flowrate.clear()
    for _ in range(measure):
        prop.propagate(sim.cars, sim.obs)
    return sum(sim.obs.flowrate)/len(sim.obs.flowrate)


def fundamental_diagram_multiL(L_list=(30, 50, 100, 200),
                               rhos=np.linspace(0.05, 0.95, 19),
                               vmax=2, p=0.5, burn_factor=10, meas_factor=5,
                               init_mode='even', tol=5e-3):
    """
    Plot q(rho) for several L. Burn-in and measure windows scale with L.
    Prints a simple finite-size deviation table vs the largest L.
    """
    flows_by_L = {}
    for L in L_list:
        burn_in = int(burn_factor * L)     # longer roads ⇒ longer equilibration
        measure = int(meas_factor * L)
        qs = []
        for rho in rhos:
            N = max(1, min(L-1, int(round(rho * L))))
            qs.append(avg_flow(N, L, vmax=vmax, p=p,
                               burn_in=burn_in, measure=measure,
                               init_mode=init_mode))
        flows_by_L[L] = np.array(qs)

    # Plot: fundamental diagrams for different L on same axes
    plt.figure()
    for L in L_list:
        plt.plot(rhos, flows_by_L[L], marker='o', ms=3, label=f'L={L}')
    plt.xlabel('density  N/L')
    plt.ylabel('flow rate')
    plt.title(f'Fundamental diagram vs system size (vmax={vmax}, p={p})')
    plt.legend(); plt.tight_layout(); plt.show()

    # Deviation from the largest-L result
    Lref = max(L_list)
    qref = flows_by_L[Lref]
    print('Max |Δq̄| versus Lref={} (over all ρ):'.format(Lref))
    for L in sorted(L_list):
        dmax = float(np.max(np.abs(flows_by_L[L] - qref)))
        flag = '✓' if dmax < tol else '✗'
        print(f'  L={L:4d}: max deviation = {dmax:.4f}  ({flag} tol={tol})')

    # Optional: show convergence at a few fixed densities
    fixed_rhos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    plt.figure()
    for rho in fixed_rhos:
        qL = []
        for L in L_list:
            # use the precomputed grid (match nearest rho index)
            idx = int(np.argmin(np.abs(rhos - rho)))
            qL.append(flows_by_L[L][idx])
        plt.plot(L_list, qL, 'o-', label=f'ρ≈{rhos[idx]:.2f}')
    plt.xlabel('road length L')
    plt.ylabel('flow rate')
    plt.title('Size convergence at Fixed densities')
    plt.legend(); plt.tight_layout(); plt.show()

    return rhos, flows_by_L


# ---------------- main() for 2.2(c) ----------------
def main():
    # (You can comment out the 2.2(b) block if you only want 2.2(c) runs.)
    # Finite-size test: when does q̄(ρ) stop depending on L?
    L_list = [30, 50, 70, 100, 200, 300, 400, 500, 600, 700]      # try more/less as you wish
    rhos, flows_by_L = fundamental_diagram_multiL(
        L_list=L_list,
        rhos=np.linspace(0.05, 0.95, 19),
        vmax=2, p=0.5,
        burn_factor=10,   # longer roads → proportionally longer burn-in
        meas_factor=5,
        init_mode='even',
        tol=5e-3          # “independent” if within 0.005 in q̄
    )

# Calling 'main()' if the script is executed.
# If the script is instead just imported, main is not called (this can be useful if you want to
# write another script importing and utilizing the functions and classes defined in this one)
if __name__ == "__main__" :
    main()

