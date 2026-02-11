#!/bin/python3

# Python simulation of a double pendulum with real time animation.
# BH, MP, AJ, TS 2020-10-27, latest version 2022-10-25.

from matplotlib import animation
import matplotlib.pyplot as plt

# Using numba to speed up force calculation
# More info: https://numba.pydata.org/numba-doc/latest/user/5minguide.html
import numba as nmb

# Numba likes numpy
import numpy as np

"""
    This script simulates and animates a double pendulum.
    Classes are similar to the ones of pendolum_template.py. The main differences are:
    - coordinates are obtained from the total energy value E (look at what functions
        Oscillator.p2squaredFromH and Oscillator.__init__ do)
    - you are asked to implement the expression for the derivatives of the Hamiltonian 
        w.r.t. coordinates p1 and p2
    - you are asked to check when the conditions to produce the Poincare' map are
        satisfied and append the coordinates' values to some container to then plot
"""

# Global constants
G = 9.8  # gravitational acceleration

"""
    The state vector is defined as:
        x = [q1, q2, p1, p2]
    That is: q1=x[0], q2=x[1], p1=x[2], p2=x[3]
"""

# Kinetic energy
def Ekin(osc):
    return 1 / (2.0 * osc.m * osc.L * osc.L) * ( osc.x[2] * osc.x[2] + 2.0 * osc.x[3] * osc.x[3] - 2.0 * osc.x[2] * osc.x[3] * np.cos(
        osc.x[0] - osc.x[1])) / (1 + (np.sin(osc.x[0] - osc.x[1])) ** 2)

# Potential energy
def Epot(osc):
    return osc.m * G * osc.L * (3 - 2 * np.cos(osc.x[0]) - np.cos(osc.x[1]))


# Class that holds the parameter and state of a double pendulum
class Oscillator:

    def p2squaredFromH(self):
        return (self.E - Epot(self)) * (1 + (np.sin(self.x[0] - self.x[1])) ** 2) * self.m * self.L * self.L

    # Initial condition is [q1, q2, p1, p2]; p2 is however re-obtained based on the value of E
    # therefore you can use any value for init_cond[3]
    def __init__(self, m=1, L=1, t0=0, E=15, init_cond=[0.0, 0.0, 0.0, -1.0]) :
        self.m = m      # mass of the pendulum bob
        self.L = L      # arm length
        self.t = t0     # the initial time
        self.E = E      # total conserved energy
        self.x = np.zeros(4)
        self.x[0] = init_cond[0]
        self.x[1] = init_cond[1]
        self.x[2] = init_cond[2]
        self.x[3] = -1.0
        while (self.x[3] < 0):
            # Comment the two following lines in case you want to exactly prescribe values to q1 and q2
            # However, be sure that there exists a value of p2 compatible with the imposed total energy E!
            self.x[0] = np.pi * (2 * np.random.random() - 1)
            self.x[1] = np.pi * (2 * np.random.random() - 1)
            p2squared = self.p2squaredFromH()
            if (p2squared >= 0):
                self.x[3] = np.sqrt(p2squared)
        self.q2_prev = self.x[1]
        print("Initialization:")
        print("E  = "+str(self.E))
        print("q1 = "+str(self.x[0]))
        print("q2 = "+str(self.x[1]))
        print("p1 = "+str(self.x[2]))
        print("p2 = "+str(self.x[3]))


# Class for storing observables for an oscillator
class Observables:

    def __init__(self):
        self.time = []          # list to store time
        self.q1list = []        # list to store q1
        self.q2list = []        # list to store q2
        self.p1list = []        # list to store p1
        self.p2list = []        # list to store p2
        self.epot = []          # list to store potential energy
        self.ekin = []          # list to store kinetic energy
        self.etot = []          # list to store total energy
        self.poincare_q1 = []   # list to store q1 for Poincare plot
        self.poincare_p1 = []   # list to store p1 for Poincare plot


# Derivate of H with respect to p1
@nmb.jit(nopython=True)
def dHdp1(x, m, L):
    q1, q2, p1, p2 = x[0], x[1], x[2], x[3]
    delta = q1 - q2
    s = np.sin(delta)
    c = np.cos(delta)
    denom = m * L * L * (1.0 + s * s)
    return (p1 - p2 * c) / denom


# Derivate of H with respect to p2
@nmb.jit(nopython=True)
def dHdp2(x, m, L):
    q1, q2, p1, p2 = x[0], x[1], x[2], x[3]
    delta = q1 - q2
    s = np.sin(delta)
    c = np.cos(delta)
    denom = m * L * L * (1.0 + s * s)
    return (2.0 * p2 - p1 * c) / denom


# Derivate of H with respect to q1
@nmb.jit(nopython=True)
def dHdq1(x, m, L):
    return 1 / (2.0 * m * L * L) * (
        -2 * (x[2] * x[2] + 2 * x[3] * x[3]) * np.cos(x[0] - x[1]) + x[2] * x[3] * (4 + 2 * (np.cos(x[0] - x[1])) ** 2)) * np.sin(
            x[0] - x[1]) / (1 + (np.sin(x[0] - x[1])) ** 2) ** 2 + m * G * L * 2.0 * np.sin(x[0])

# Derivate of H with respect to q2
@nmb.jit(nopython=True)
def dHdq2(x, m, L):
    return 1 / (2.0 * m * L * L) * (
        2 * (x[2] * x[2] + 2 * x[3] * x[3]) * np.cos(x[0] - x[1]) - x[2] * x[3] * (4 + 2 * (np.cos(x[0] - x[1])) ** 2)) * np.sin(x[0] - x[1]) / (
            1 + (np.sin(x[0] - x[1])) ** 2) ** 2 + m * G * L * np.sin(x[1])

def HamiltonEquations(x, m, L):
    return np.array([dHdp1(x, m, L), dHdp2(x, m, L), -dHdq1(x, m, L), -dHdq2(x, m, L)])

class RK4Integrator:

    def __init__(self, dt=0.003):
        self.dt = dt    # time step
        self.mult_vec = np.transpose(np.array([1,2,2,1]))

    def integrate(self,
                  osc,
                  obs, 
                  ):

        """ Perform a single integration step """
        self.timestep(osc, obs)

        """ Append observables to their lists """
        obs.time.append(osc.t)
        obs.q1list.append(osc.x[0])
        obs.q2list.append(osc.x[1])
        obs.p1list.append(osc.x[2])
        obs.p2list.append(osc.x[3])
        obs.epot.append(Epot(osc))
        obs.ekin.append(Ekin(osc))
        obs.etot.append(Epot(osc) + Ekin(osc))
        # TODO: Append values for the Poincare map

        # q2 crosses 0 from below AND (p2 - p1) > 0 ---
        if len(obs.q2list) >= 2:
            if (obs.q2list[-2] < 0.0) and (obs.q2list[-1] >= 0.0):
                if (obs.p2list[-1] - obs.p1list[-1]) > 0.0:
                    obs.poincare_q1.append(obs.q1list[-1])
                    obs.poincare_p1.append(obs.p1list[-1])

    """
        Implementation of RK4 for a system of 4 variables
        It's much more compact when you write in in vector form
    """
    def timestep(self, osc, obs):
        dt = self.dt
        osc.t += dt

        # Initialization
        x = osc.x
        m = osc.m
        L = osc.L

        # RK4 coefficients (Butcher tableau)
        ab = np.zeros( (4,4), dtype=float )

        # First sub-step:
        ab[:,0] = dt*HamiltonEquations(x,m,L)

        # Second sub-step:
        ab[:,1] = dt*HamiltonEquations(x+0.5*ab[:,0],m,L)

        # Third sub-step:
        ab[:,2] = dt*HamiltonEquations(x+0.5*ab[:,1],m,L)

        # Fourth sub-step:
        ab[:,3] = dt*HamiltonEquations(x+ab[:,2],m,L)

        osc.x += np.matmul(ab,self.mult_vec) / 6.0


# Animation function which integrates a few steps and return a line for the pendulum
def animate(framenr, osc, obs, integrator, pendulum_lines, stepsperframe):
    for it in range(stepsperframe):
        integrator.integrate(osc, obs)

    x1 = np.sin(osc.x[0])
    y1 = -np.cos(osc.x[0])
    x2 = x1 + np.sin(osc.x[1])
    y2 = y1 - np.cos(osc.x[1])
    pendulum_lines.set_data([0, x1, x2], [0, y1, y2])
    return pendulum_lines,


class Simulation:

    def reset(self, osc=Oscillator()) :
        self.oscillator = osc
        self.obs = Observables()

    def __init__(self, osc=Oscillator()) :
        self.reset(osc)

    def run(self,
            integrator,
            tmax=30.,   # final time
            outfile='energy1.pdf'
            ):

        n = int(tmax / integrator.dt)

        for it in range(n):
            integrator.integrate(self.oscillator, self.obs)

        #self.plot_observables(title="Energy="+str(self.oscillator.E))

    def run_animate(self,
            integrator,
            tmax=30.,           # final time
            stepsperframe=5,    # how many integration steps between visualising frames
            outfile='energy1.pdf'
            ):

        numframes = int(tmax / (stepsperframe * integrator.dt))

        plt.clf()

        # If you experience problems visualizing the animation try to comment/uncomment this line
        fig = plt.figure()

        ax = plt.subplot(xlim=(-2.2, 2.2), ylim=(-2.2, 2.2))
        plt.axhline(y=0)  # draw a default hline at y=1 that spans the xrange
        plt.axvline(x=0)  # draw a default vline at x=1 that spans the yrange
        pendulum_lines, = ax.plot([], [], lw=5)

        # Call the animator, blit=True means only re-draw parts that have changed
        anim = animation.FuncAnimation(fig, animate,  # init_func=init,
                                       fargs=[self.oscillator, self.obs, integrator, pendulum_lines, stepsperframe],
                                       frames=numframes, interval=25, blit=True, repeat=False)
        
        # If you experience problems visualizing the animation try to comment/uncomment this line 
        # plt.show()

        # If you experience problems visualizing the animation try to comment/uncomment this line 
        plt.waitforbuttonpress(10)

    # Plot coordinates and energies (to be called after running)
    def plot_observables(self, title="Double pendulum") :

        plt.figure()
        plt.title(title)
        plt.xlabel('q1')
        plt.ylabel('p1')
        plt.plot(self.obs.q1list, self.obs.p1list)
        plt.tight_layout()  # adapt the plot area tot the text with larger fonts 

        plt.figure()
        plt.title(title)
        plt.xlabel('q2')
        plt.ylabel('p2')
        plt.plot(self.obs.q2list, self.obs.p2list)
        plt.plot([0.0, 0.0], [min(self.obs.p2list), max(self.obs.p2list)], 'k--')
        plt.tight_layout()  # adapt the plot area tot the text with larger fonts 

        plt.figure()
        plt.title(title)
        plt.xlabel('q1')
        plt.ylabel('p1')
        plt.plot(self.obs.poincare_q1, self.obs.poincare_p1, 'ro')
        plt.tight_layout()  # adapt the plot area tot the text with larger fonts

        plt.figure()
        plt.title(title)
        plt.xlabel('time')
        plt.ylabel('energy')
        plt.plot(self.obs.time, self.obs.epot, self.obs.time, self.obs.ekin, self.obs.time, self.obs.etot)
        plt.legend(('Epot', 'Ekin', 'Etot'))
        plt.tight_layout()  # adapt the plot area tot the text with larger fonts 

        plt.show()

 
def exercise_15a() :
    """
    np.random.seed()                 # reproducible random q1,q2
    energies = [1, 15, 40]
    dt = 0.003
    tmax = 30
    stepsperframe = 5                 # smoother animation without being too slow

    for E in energies:
        osc = Oscillator(m=1.0, L=1.0, t0=0.0, E=E, init_cond=[0.0, 0.0, 0.0, -1.0])
        sim = Simulation(osc)
        integ = RK4Integrator(dt)

        # animate
        # sim.run_animate(integrator=integ, tmax=tmax, stepsperframe=stepsperframe)
        sim.run(integrator=integ, tmax=tmax)

        # plot diagnostics (q–p planes and energies)
        #sim.plot_observables(title=f"Energy = {E:g}")
    """
    """
    E = 15.0
    dt = 0.003
    tmax = 30000                      # a bit longer to densify the section
    initials = [(0.0, 0.0, 0.0),     # (q1, q2, p1)
                (1.1, 0.0, 0.0)]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    for (q1_0, q2_0, p1_0) in initials:
        # build oscillator, then enforce the desired IC and recompute p2
        osc = Oscillator(m=1.0, L=1.0, t0=0.0, E=E, init_cond=[0.0, 0.0, 0.0, -1.0])
        osc.x[0] = q1_0
        osc.x[1] = q2_0
        osc.x[2] = p1_0               # p1 = 0 in the figure
        p2sq = osc.p2squaredFromH()   # valid when p1 = 0

        osc.x[3] = np.sqrt(p2sq)      # pick the positive branch; then (p2 - p1) > 0 initially

        sim = Simulation(osc)
        integ = RK4Integrator(dt)
        sim.run(integrator=integ, tmax=tmax)   # fills obs.poincare_* via the hook above

        ax.plot(sim.obs.poincare_q1, sim.obs.poincare_p1, '.',
                ms=0.8, alpha=0.6, label=f"IC q1={q1_0:g}, q2={q2_0:g}")

    ax.set_title("Poincaré map (E = 15),  q2 = 0,  p2 − p1 > 0")
    ax.set_xlabel("q1")
    ax.set_ylabel("p1")
    ax.set_xlim(-1.6, 1.6)           # roughly like Fig. 6.13
    ax.set_ylim(-7.0, 8.5)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, ls=':', alpha=0.4)
    plt.tight_layout()
    plt.show()     
    """

    np.random.seed()         # reproducible random ICs
    energies = [12]
    n_ic     = 10
    dt       = 0.003
    tmax     = 200          # a bit longer so each IC contributes enough hits

    for E in energies:
        fig, ax = plt.subplots(figsize=(7, 5.0))
        for _ in range(n_ic):
            osc   = Oscillator(E=E)
            sim   = Simulation(osc)
            integ = RK4Integrator(dt)

            nsteps = int(tmax/dt)
            for __ in range(nsteps):
                integ.integrate(sim.oscillator, sim.obs)

            ax.plot(sim.obs.poincare_q1, sim.obs.poincare_p1, '.', ms=0.9, alpha=0.6)

        ax.set_title(f"Poincaré map (E = {E:g})")
        ax.set_xlabel('q1')
        ax.set_ylabel('p1')
        ax.grid(True, ls=':')
        plt.tight_layout()
        plt.show()
    
if __name__ == "__main__" :
    exercise_15a()
    # exercise_15b()
    # ...