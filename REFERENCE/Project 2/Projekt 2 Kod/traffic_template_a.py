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
    




# ---- Minimal helpers for 2.2(a) ----

def run_avg_flow(numCars, roadLength, vmax=2, p=0.5, burn_in=200, measure=500, v0=0):
    """Return time-averaged flow after burn-in for one (N,L)."""
    cars = Cars(numCars=numCars, roadLength=roadLength, v0=v0)
    sim  = Simulation(cars)
    prop = MyPropagator(vmax=vmax, p=p)

    # 1) Burn-in (do not record)
    for _ in range(burn_in):
        prop.propagate(sim.cars, sim.obs)

    # 2) Clear obs and measure flow over a window
    sim.obs.time.clear()
    sim.obs.flowrate.clear()
    for _ in range(measure):
        prop.propagate(sim.cars, sim.obs)

    # 3) Time-averaged flow
    return sum(sim.obs.flowrate) / len(sim.obs.flowrate)

def fundamental_diagram(roadLength=100, vmax=2, p=0.5, burn_in=300, measure=800):
    rhos, flows = [], []
    for numCars in range(1, roadLength):           # densities in (0,1)
        qbar = run_avg_flow(numCars, roadLength, vmax, p, burn_in, measure)
        rhos.append(numCars / roadLength)
        flows.append(qbar)

    # Plot q vs rho
    plt.figure()
    plt.plot(rhos, flows, 'o-')
    plt.xlabel('density N/L')
    plt.ylabel('time-averaged flow')
    plt.title(f'Fundamental diagram (L={roadLength}, vmax={vmax}, p={p})')
    plt.tight_layout()
    plt.show()

    # Report density where jams begin (peak location)
    rho_c = rhos[int(np.argmax(flows))]
    print(f'Peak flow at density ρ* ≈ {rho_c:.3f} (onset of jams beyond this).')

    return rhos, flows






# It's good practice to encapsulate the script execution in 
# a main() function (e.g. for profiling reasons)
def main() :

    # Here you can define one or more instances of cars, with possibly different parameters, 
    # and pass them to the simulator 

    # Be sure you are passing the correct initial conditions!
    cars = Cars(numCars = 5, roadLength=50)

    # Create the simulation object for your cars instance:
    simulation = Simulation(cars)

    # simulation.run_animate(propagator=ConstantPropagator())
    simulation.run_animate(propagator=MyPropagator(vmax=2, p=0.5))




    fundamental_diagram(roadLength=100, vmax=2, p=0.5, burn_in=300, measure=800)


# Calling 'main()' if the script is executed.
# If the script is instead just imported, main is not called (this can be useful if you want to
# write another script importing and utilizing the functions and classes defined in this one)
if __name__ == "__main__" :
    main()

