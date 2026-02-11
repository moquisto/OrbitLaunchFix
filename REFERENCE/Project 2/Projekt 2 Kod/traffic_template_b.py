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
    






# ====== 2.2(b) helpers ======

def avg_flow_fixed_setup(burn_in=300, measure=100, init_mode='even', seed=None):
    """
    One independent simulation at L=50, N=25, vmax=2, p=0.5.
    Returns time-averaged flow over 'measure' steps after 'burn_in'.
    init_mode: 'even' (default) or 'jam' (clustered) or 'random' (see Cars init below).
    """
    if seed is not None:
        rng.seed(seed)
    else:
        rng.seed(None)  # re-seed from OS for independence

    # Allow alternate initializations without breaking old code:
    cars = Cars(numCars=25, roadLength=50, v0=0)          # your existing init (even spacing)
    if init_mode == 'jam':
        # compact cluster 0..N-1 (keep arrays ordered)
        cars.x = list(range(cars.numCars))
        cars.v = [0]*cars.numCars
    elif init_mode == 'random':
        # choose N distinct sites uniformly, then sort so order matches road order
        pos = sorted(rng.choice(cars.roadLength, size=cars.numCars, replace=False).tolist())
        cars.x = pos
        cars.v = [0]*cars.numCars

    sim  = Simulation(cars)
    prop = MyPropagator(vmax=2, p=0.5)

    # burn-in (not recorded)
    for _ in range(burn_in):
        prop.propagate(sim.cars, sim.obs)

    # measure average flow for 'measure' steps
    sim.obs.time.clear(); sim.obs.flowrate.clear()
    for _ in range(measure):
        prop.propagate(sim.cars, sim.obs)

    return sum(sim.obs.flowrate)/len(sim.obs.flowrate)


def se_of_mean(samples):
    """Return (mean, standard error) for a list of numbers."""
    n = len(samples)
    if n <= 1:
        return (samples[0] if n else float('nan'), float('inf'))
    mean = float(np.mean(samples))
    se   = float(np.std(samples, ddof=1) / np.sqrt(n))
    return mean, se


def runs_needed_for_target_se(target_se=1e-3, burn_in=300, measure=100, init_mode='even',
                              max_runs=10000):
    """
    Keep adding independent runs until SE <= target_se (or max_runs reached).
    Returns (runs_used, mean_flow, se, history_of_se) for plotting/diagnostics.
    """
    flows = []
    se_hist = []
    for m in range(1, max_runs+1):
        flows.append(avg_flow_fixed_setup(burn_in, measure, init_mode))
        mean, se = se_of_mean(flows)
        se_hist.append(se)
        if m >= 2 and se <= target_se:
            return m, mean, se, se_hist
    return max_runs, *se_of_mean(flows), se_hist


def se_curve(max_runs=1000, burn_in=300, measure=100, init_mode='even'):
    """SE vs number of runs (1..max_runs) — useful to visualize convergence."""
    flows, hist = [], []
    for m in range(1, max_runs+1):
        flows.append(avg_flow_fixed_setup(burn_in, measure, init_mode))
        _, se = se_of_mean(flows)
        hist.append(se)
    return hist


def burn_in_scan(burns=(0, 50, 100, 200, 400, 800, 1200), runs_per_b=200, init_mode='even'):
    """
    For each burn-in length, do 'runs_per_b' independent estimates and return
    lists of (burn_in, mean_flow, SE). Lets you see how much burn-in is needed
    and whether results depend on init_mode.
    """
    B, means, ses = [], [], []
    for b in burns:
        flows = [avg_flow_fixed_setup(burn_in=b, measure=100, init_mode=init_mode)
                 for _ in range(runs_per_b)]
        mean, se = se_of_mean(flows)
        B.append(b); means.append(mean); ses.append(se)
    return B, means, ses








def main():
    # --- Optional quick visual sanity check (not required for 2.2b) ---
    # cars = Cars(numCars=25, roadLength=50, v0=0)
    # Simulation(cars).run_animate(propagator=MyPropagator(vmax=2, p=0.5),
    #                              numsteps=200, stepsperframe=2, title="preview")

    # === 2.2(b): statistical accuracy at L=50, N=25, vmax=2, p=0.5 ===
    TARGET_SE = 1e-3
    BURN_IN   = 300          # you will probe this below
    MEASURE   = 100          # exactly as the task states

    # 1) How many simulations to reach SE = 0.001? (even vs jam init)
    m_even, mean_even, se_even, se_hist_even = runs_needed_for_target_se(
        TARGET_SE, burn_in=BURN_IN, measure=MEASURE, init_mode='even', max_runs=5000)
    m_jam,  mean_jam,  se_jam,  se_hist_jam  = runs_needed_for_target_se(
        TARGET_SE, burn_in=BURN_IN, measure=MEASURE, init_mode='jam',  max_runs=5000)

    print(f"[even init]  runs needed: {m_even}, mean flow={mean_even:.5f}, SE={se_even:.5f}")
    print(f"[jam  init]  runs needed: {m_jam},  mean flow={mean_jam:.5f},  SE={se_jam:.5f}")

    # Plot: SE vs number of simulations (convergence)
    plt.figure()
    plt.plot(range(1, len(se_hist_even)+1), se_hist_even, label='even init')
    plt.plot(range(1, len(se_hist_jam)+1),  se_hist_jam,  label='jam init')
    plt.axhline(TARGET_SE, ls='--', color='k', label='target SE=0.001')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('number of simulations M')
    plt.ylabel('standard error of mean flow')
    plt.title('standard error vs M with L=50, N=25, vmax=2, p=0.5')
    plt.legend(); plt.tight_layout(); plt.show()

    # 2) How long does equilibration need to be?  (and does it depend on init?)
    burns = (0, 10, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500)
    B_e, mean_e, se_e = burn_in_scan(burns, runs_per_b=200, init_mode='even')
    B_j, mean_j, se_j = burn_in_scan(burns, runs_per_b=200, init_mode='jam')

    # Plot: estimated mean flow vs burn-in for both initializations (with SE bands)
    plt.figure()
    plt.errorbar(B_e, mean_e, yerr=se_e, fmt='o-', capsize=3, label='even init')
    plt.errorbar(B_j, mean_j, yerr=se_j, fmt='s--', capsize=3, label='jam init')
    plt.xlabel('burn in steps (discarded steps)')
    plt.ylabel('time averaged flow over 100 steps')
    plt.title('Effect of burn in and initial conditions on flow estimate')
    plt.legend(); plt.tight_layout(); plt.show()

    # A compact textual answer you can paste in your report:
    best_b = B_e[int(np.argmin(se_e))]
    print(f"Suggested burn-in ≈ {best_b}–{max(B_e)} steps (flow, SE stabilize there). "
          f"Initialization matters: jammed starts typically need longer burn-in.")
    



# Calling 'main()' if the script is executed.
# If the script is instead just imported, main is not called (this can be useful if you want to
# write another script importing and utilizing the functions and classes defined in this one)
if __name__ == "__main__" :
    main()

