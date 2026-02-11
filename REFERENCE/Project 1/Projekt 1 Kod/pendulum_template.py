#!/bin/python3

# Python simulation of a simple planar pendulum with real time animation
# BH, OF, MP, AJ, TS 2020-10-20, latest version 2022-10-25.

from matplotlib import animation
from pylab import *
import numpy as np
import matplotlib.pyplot as plt

"""
    This script defines all the classes needed to simulate (and animate) a single pendulum.
    Hierarchy (somehow in order of encapsulation):
    - Oscillator: a struct that stores the parameters of an oscillator (harmonic or pendulum)
    - Observable: a struct that stores the oscillator's coordinates and energy values over time
    - BaseSystem: harmonic oscillators and pendolums are distinguished only by the expression of
                    the return force. This base class defines a virtual force method, which is
                    specified by its child classes
                    -> Harmonic: specifies the return force as -k*t (i.e. spring)
                    -> Pendulum: specifies the return force as -k*sin(t)
    - BaseIntegrator: parent class for all time-marching schemes; function integrate performs
                    a numerical integration steps and updates the quantity of the system provided
                    as input; function timestep wraps the numerical scheme itself and it's not
                    directly implemented by BaseIntegrator, you need to implement it in his child
                    classes (names are self-explanatory)
                    -> EulerCromerIntegrator: ...
                    -> VerletIntegrator: ...
                    -> RK4Integrator: ...
    - Simulation: this last class encapsulates the whole simulation procedure; functions are 
                    self-explanatory; you can decide whether to just run the simulation or to
                    run while also producing an animation: the latter option is slower
"""

# Global constants
G = 9.8  # gravitational acceleration

class Oscillator:

    """ Class for a general, simple oscillator """

    def __init__(self, m=1, c=4, t0=0, theta0=0, dtheta0=0, gamma=0):
        self.m = m              # mass of the pendulum bob
        self.c = c              # c = g/L
        self.L = G / c          # string length
        self.t = t0             # the time
        self.theta = theta0     # the position/angle
        self.dtheta = dtheta0   # the velocity
        self.gamma = gamma      # damping

class Observables:

    """ Class for storing observables for an oscillator """

    def __init__(self):
        self.time = []          # list to store time
        self.pos = []           # list to store positions
        self.vel = []           # list to store velocities
        self.energy = []        # list to store energy


class BaseSystem:
    
    def force(self, osc):

        """ Virtual method: implemented by the childc lasses  """

        pass


class Harmonic(BaseSystem):
    def force(self, osc):
        return - osc.m * ( osc.c*osc.theta + osc.gamma*osc.dtheta )


class Pendulum(BaseSystem):
    def force(self, osc):
        return - osc.m * ( osc.c*np.sin(osc.theta) + osc.gamma*osc.dtheta )


class BaseIntegrator:

    def __init__(self, _dt=0.01) :
        self.dt = _dt   # time step

    def integrate(self, simsystem, osc, obs):

        """ Perform a single integration step """
        
        self.timestep(simsystem, osc, obs)

        # Append observables to their lists
        obs.time.append(osc.t)
        obs.pos.append(osc.theta)
        obs.vel.append(osc.dtheta)
        # Function 'isinstance' is used to check if the instance of the system object is 'Harmonic' or 'Pendulum'
        if isinstance(simsystem, Harmonic) :
            # Harmonic oscillator energy
            obs.energy.append(0.5 * osc.m * (osc.L ** 2) * (osc.dtheta ** 2) + 0.5 * osc.m * G * osc.L * (osc.theta ** 2))
        else :
            # Pendulum energy
            obs.energy.append(0.5*osc.m*(osc.L**2)*(osc.dtheta**2) + osc.m*G*osc.L*(1 - np.cos(osc.theta)))
            # TODO: Append the total energy for the pendulum (use the correct formula!)

    def timestep(self, simsystem, osc, obs):

        """ Virtual method: implemented by the child classes """
        
        pass


# HERE YOU ARE ASKED TO IMPLEMENT THE NUMERICAL TIME-MARCHING SCHEMES:

class EulerCromerIntegrator(BaseIntegrator):
    def timestep(self, simsystem, osc, obs):
        accel = simsystem.force(osc) / osc.m
        osc.t += self.dt
        # TODO: Implement the integration here, updating osc.theta and osc.dtheta

        # v_{n+1} = v_n + a_n Δt
        osc.dtheta += accel * self.dt
        # x_{n+1} = x_n + v_{n+1} Δt
        osc.theta += osc.dtheta * self.dt


class VerletIntegrator(BaseIntegrator):
    def timestep(self, simsystem, osc, obs):
        accel = simsystem.force(osc) / osc.m
        osc.t += self.dt
        # TODO: Implement the integration here, updating osc.theta and osc.dtheta

        osc.theta += osc.dtheta*self.dt + 0.5*accel*(self.dt**2)

        accel_new = simsystem.force(osc) / osc.m

        osc.dtheta += 0.5*(accel_new + accel) * self.dt


import copy

class RK4Integrator(BaseIntegrator):
    def timestep(self, simsystem, osc, obs):
        # TODO: Implement the integration here, updating osc.theta and osc.dtheta
        
        dt = self.dt

        # k1
        k1_theta = osc.dtheta
        k1_omega = simsystem.force(osc) / osc.m

        # k2
        o = copy.deepcopy(osc)
        o.theta  += 0.5*dt*k1_theta
        o.dtheta += 0.5*dt*k1_omega
        k2_theta = o.dtheta
        k2_omega = simsystem.force(o) / o.m

        # k3
        o = copy.deepcopy(osc)
        o.theta  += 0.5*dt*k2_theta
        o.dtheta += 0.5*dt*k2_omega
        k3_theta = o.dtheta
        k3_omega = simsystem.force(o) / o.m

        # k4
        o = copy.deepcopy(osc)
        o.theta  += dt*k3_theta
        o.dtheta += dt*k3_omega
        k4_theta = o.dtheta
        k4_omega = simsystem.force(o) / o.m

        # combine
        osc.theta  += (dt/6.0)*(k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
        osc.dtheta += (dt/6.0)*(k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
        osc.t      += dt
    

# Animation function which integrates a few steps and return a line for the pendulum
def animate(framenr, simsystem, oscillator, obs, integrator, pendulum_line, stepsperframe):
    
    for it in range(stepsperframe):
        integrator.integrate(simsystem, oscillator, obs)

    x = np.array([0, np.sin(oscillator.theta)])
    y = np.array([0, -np.cos(oscillator.theta)])
    pendulum_line.set_data(x, y)
    return pendulum_line,


class Simulation:

    def reset(self, osc=Oscillator()) :
        self.oscillator = osc
        self.obs = Observables()

    def __init__(self, osc=Oscillator()) :
        self.reset(osc)

    # Run without displaying any animation (fast)
    def run(self,
            simsystem,
            integrator,
            tmax=30.,               # final time
            ):

        n = int(tmax / integrator.dt)

        for it in range(n):
            integrator.integrate(simsystem, self.oscillator, self.obs)

    # Run while displaying the animation of a pendulum swinging back and forth (slow-ish)
    # If too slow, try to increase stepsperframe
    def run_animate(self,
            simsystem,
            integrator,
            tmax=30.,               # final time
            stepsperframe=1         # how many integration steps between visualising frames
            ):

        numframes = int(tmax / (stepsperframe * integrator.dt))-2

        # WARNING! If you experience problems visualizing the animation try to comment/uncomment this line
        plt.clf()

        # If you experience problems visualizing the animation try to comment/uncomment this line
        # fig = plt.figure()

        ax = plt.subplot(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
        plt.axhline(y=0)  # draw a default hline at y=1 that spans the xrange
        plt.axvline(x=0)  # draw a default vline at x=1 that spans the yrange
        pendulum_line, = ax.plot([], [], lw=5)
        plt.title(title)
        # Call the animator, blit=True means only re-draw parts that have changed
        anim = animation.FuncAnimation(plt.gcf(), animate,  # init_func=init,
                                       fargs=[simsystem,self.oscillator,self.obs,integrator,pendulum_line,stepsperframe],
                                       frames=numframes, interval=25, blit=True, repeat=False)

        # If you experience problems visualizing the animation try to comment/uncomment this line
        # plt.show()

        # If you experience problems visualizing the animation try to comment/uncomment this line
        plt.waitforbuttonpress(10)

    # Plot coordinates and energies (to be called after running)
    def plot_observables(self, title="simulation", ref_E=None):

        plt.clf()
        plt.title(title)
        plt.plot(self.obs.time, self.obs.pos, 'b-', label="Position")
        plt.plot(self.obs.time, self.obs.vel, 'r-', label="Velocity")
        plt.plot(self.obs.time, self.obs.energy, 'g-', label="Energy")
        if ref_E != None :
            plt.plot([self.obs.time[0],self.obs.time[-1]] , [ref_E, ref_E], 'k--', label="Ref.")
        plt.xlabel('time')
        plt.ylabel('observables')
        plt.legend()
        plt.savefig(title + ".pdf")
        plt.show()


# It's good practice to encapsulate the script execution in 
# a function (e.g. for profiling reasons)
def exercise_11():
    """ 1.1
    m = 1
    omega = 2
    c = omega**2          # ho
    gamma = 0
    time_steps = [0.01, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.20]

    integrators = [
        ("EulerCromer", EulerCromerIntegrator),
        ("Verlet",      VerletIntegrator),
        ("RK4",         RK4Integrator),
    ]

    method_color = {
        "EulerCromer": "#0099ff",
        "Verlet":      "#00ff00",
        "RK4":         "#ff0000",
    }

    mark = {
        0.1*np.pi: "o",
        0.5*np.pi: "s",
    }

    # Harmonic oscillator: exact solution comparison
    x0, v0 = 1, 0
    tmax_list_ho = [30, 30000]

    # collect errors for both horizons
    max_error_by_method = {tmax_value: {name: [] for name, _ in integrators}
                           for tmax_value in tmax_list_ho}

    for tmax_value in tmax_list_ho:
        for dt in time_steps:
            for method_name, IntegratorClass in integrators:
                oscillator = Oscillator(m=m, c=c, t0=0, theta0=x0, dtheta0=v0, gamma=gamma)
                simulation = Simulation(oscillator)
                simulation.run(Harmonic(), IntegratorClass(dt), tmax=tmax_value)

                time_array = np.array(simulation.obs.time)
                x_exact = x0*np.cos(omega*time_array) + (v0/omega)*np.sin(omega*time_array)
                x_num   = np.array(simulation.obs.pos)
                max_error = np.max(np.abs(x_num - x_exact))   # RAW max error
                max_error_by_method[tmax_value][method_name].append(max_error)

    # single figure with two subplots (30 s, 30000 s)
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True, sharey=True)
    for ax, tmax_value in zip(axes, tmax_list_ho):
        for method_name, _ in integrators:
            ax.loglog(
                time_steps,
                max_error_by_method[tmax_value][method_name],
                marker="o",
                linestyle="-",
                linewidth=1.8,
                markersize=5,
                color=method_color[method_name],
                label=method_name
            )
        ax.grid(True, which='both', ls=':')
        ax.set_ylabel(r'Max position error $|x - x_{\mathrm{exact}}|$')
        ax.set_title(f'Harmonic oscillator: error vs $\Delta t$ (tmax = {tmax_value:g} s)')
    axes[-1].set_xlabel(r'$\Delta t$ [s]')
    axes[0].legend()
    plt.tight_layout()
    plt.savefig('HO_max_error_vs_dt_RAW_combined.pdf')
    plt.show()

    # Pendulum: raw energy error vs deltat
    c_pend = 4.0
    theta0_list = [0.1*np.pi, 0.5*np.pi]
    tmax_list   = [30, 30000]

    # collect raw max energy errors
    raw_energy_error = {tmax_value: {name: {theta0: [] for theta0 in theta0_list}
                        for (name, _) in integrators} for tmax_value in tmax_list}

    for tmax_value in tmax_list:
        for dt in time_steps:
            for method_name, IntegratorClass in integrators:
                for theta0 in theta0_list:
                    oscillator = Oscillator(m=m, c=c_pend, t0=0, theta0=theta0, dtheta0=0.0, gamma=gamma)
                    simulation = Simulation(oscillator)
                    simulation.run(Pendulum(), IntegratorClass(dt), tmax=tmax_value)

                    # exact pendulum energy at t=0 (dθ0=0)
                    E0 = m*G*oscillator.L*(1 - np.cos(theta0))
                    energy_series  = np.array(simulation.obs.energy)
                    dE = np.max(np.abs(energy_series - E0))           # raw max energy error
                    raw_energy_error[tmax_value][method_name][theta0].append(dE)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True, sharey=True)
    for ax, tmax_value in zip(axes, tmax_list):
        for method_name, _ in integrators:
            for theta0 in theta0_list:
                ax.loglog(
                    time_steps,
                    raw_energy_error[tmax_value][method_name][theta0],
                    marker=mark[theta0],
                    linestyle=':' if np.isclose(theta0/np.pi, 0.5) else '-',  # dotted for 0.5π
                    linewidth=1.8,
                    markersize=5,
                    color=method_color[method_name],
                    label=f"{method_name}, θ0/π={theta0/np.pi:.1f}"
                )
        ax.grid(True, which='both', ls=':')
        ax.set_ylabel(r'Max energy error $|E(t)-E(0)|$ J')
        ax.set_title(f'Pendulum: energy error vs Δt (tmax = {tmax_value:g} s)')
    axes[-1].set_xlabel(r'$\Delta t$ [s]')

    handles, labels = axes[0].get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    axes[0].legend(uniq.values(), uniq.keys(), fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig('pendulum_energy_error_vs_dt_combined_RAW_colors.pdf')
    plt.show()

    # 3) Chosen method rk4
    chosen_integrator = RK4Integrator
    chosen_dt = 0.02
    for theta0 in theta0_list:
        oscillator = Oscillator(m=m, c=c_pend, t0=0, theta0=theta0, dtheta0=0.0, gamma=gamma)
        simulation = Simulation(oscillator)
        simulation.run(Pendulum(), chosen_integrator(chosen_dt), tmax=30.0)

        # reference energy line for the plot (pendulum exact at t=0)
        E0 = m*G*oscillator.L*(1 - np.cos(theta0))
        simulation.plot_observables(
            title=f'pendulum, {chosen_integrator.__name__}, theta{theta0/np.pi:.1f}, dt{chosen_dt:.2f}',
            ref_E=E0
        )
    """



    """ 1.2

    step_size = 1e-3                  
    total_time_seconds = 30           
    discard_cycles = 1                # discard first measured cycle before averaging
    mass = 1
    c_value = 4                     
    damping = 0

    # angles: 20 values from 0 to 2π
    theta0_values = np.linspace(0, 2.0*np.pi, 40, endpoint=True)

    line_width = 1.8
    marker_size = 6

    # measure period from maxima 
    def measure_period_from_velocity_peaks(time_array, velocity_array, discard_n=1):
        
        #Detect times where dtheta crosses 0 with positive->negative slope (maxima of theta).
        #Linear interpolation for sub-step accuracy. Return average period after discarding
        #the first 'discard_n' cycles; return np.nan if not enough cycles were found.
        
        max_times = []
        for i in range(1, len(velocity_array)):
            v_prev, v_next = velocity_array[i-1], velocity_array[i]
            if (v_prev > 0) and (v_next <= 0):
                t_prev, t_next = time_array[i-1], time_array[i]
                if v_next != v_prev:
                    t_cross = t_prev + (0 - v_prev) * (t_next - t_prev) / (v_next - v_prev)
                else:
                    t_cross = t_prev
                max_times.append(t_cross)

        if len(max_times) < discard_n + 2:
            return np.nan
        periods = np.diff(max_times)
        return float(np.mean(periods[discard_n:]))

    # peiods 
    theta0_over_pi = []
    period_harmonic_numeric = []
    period_harmonic_theory = []
    period_pendulum_numeric = []
    period_series = []

    # theoretical HO period 
    length_value = G / c_value
    period_ho_theory_scalar = 2.0*np.pi*np.sqrt(length_value / G)   # = 2π/√c = π for c=4

    for theta0 in theta0_values:
        # harmonic oscillator
        ho_oscillator = Oscillator(m=mass, c=c_value, t0=0,
                                   theta0=theta0, dtheta0=0, gamma=damping)
        ho_simulation = Simulation(ho_oscillator)
        ho_simulation.run(Harmonic(), RK4Integrator(step_size), tmax=total_time_seconds)

        time_array_ho = np.array(ho_simulation.obs.time)
        velocity_array_ho = np.array(ho_simulation.obs.vel)
        period_ho_numeric = measure_period_from_velocity_peaks(
            time_array_ho, velocity_array_ho, discard_n=discard_cycles
        )

        # pendulum
        pendulum_oscillator = Oscillator(m=mass, c=c_value, t0=0,
                                         theta0=theta0, dtheta0=0, gamma=damping)
        pendulum_simulation = Simulation(pendulum_oscillator)
        pendulum_simulation.run(Pendulum(), RK4Integrator(step_size), tmax=total_time_seconds)

        time_array_p = np.array(pendulum_simulation.obs.time)
        velocity_array_p = np.array(pendulum_simulation.obs.vel)
        period_p_numeric = measure_period_from_velocity_peaks(
            time_array_p, velocity_array_p, discard_n=discard_cycles
        )

        # perturbation series
        theta_val = theta0
        period_series_value = period_ho_theory_scalar * (
            1.0 + (1.0/16.0)*theta_val**2 + (11.0/3072.0)*theta_val**4 + (173.0/737280.0)*theta_val**6
        )

        # collect
        theta0_over_pi.append(theta0/np.pi)
        period_harmonic_numeric.append(period_ho_numeric)
        period_harmonic_theory.append(period_ho_theory_scalar)
        period_pendulum_numeric.append(period_p_numeric)
        period_series.append(period_series_value)

        print(f"theta0/pi={theta0/np.pi:5.2f} | T_HO_num={period_ho_numeric!s:>8} | "
              f"T_HO_th={period_ho_theory_scalar:6.3f} | T_pend_num={period_p_numeric!s:>8}")

    # to arrays
    theta0_over_pi = np.array(theta0_over_pi)
    period_harmonic_numeric = np.array(period_harmonic_numeric, dtype=float)
    period_harmonic_theory = np.array(period_harmonic_theory, dtype=float)
    period_pendulum_numeric = np.array(period_pendulum_numeric, dtype=float)
    period_series = np.array(period_series, dtype=float)

    # PLot 1
    plt.figure(figsize=(7.8, 4.8))
    plt.plot(theta0_over_pi, period_harmonic_numeric, 's-', linewidth=line_width, markersize=marker_size,
             label='Harmonic (numeric RK4)')
    plt.plot(theta0_over_pi, period_pendulum_numeric, 'o-', linewidth=line_width, markersize=marker_size,
             label='Pendulum (numeric RK4)')
    plt.axhline(period_ho_theory_scalar, color='0.3', linestyle='--', linewidth=1.6, label='Harmonic (theory)')
    plt.xlabel('theta(0) / pi')
    plt.ylabel('Period T [s]')
    plt.title('Period vs initial angle (20 values from 0 to 2π)')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig('period_vs_theta_0_to_2pi_RK4.pdf')
    plt.show()

    # Plot 2
    plt.figure(figsize=(7.8, 4.8))
    plt.plot(theta0_over_pi, period_pendulum_numeric, 'o-', linewidth=line_width, markersize=marker_size,
             label='Pendulum (numeric RK4)')
    plt.plot(theta0_over_pi, period_series, '--', linewidth=line_width,
             label='Perturbation series (to theta^6)')
    plt.axhline(period_ho_theory_scalar, color='0.3', linestyle=':', linewidth=1.6, label='Harmonic (numerical)')
    plt.xlabel('theta(0) / pi')
    plt.ylabel('Period T [s]')
    plt.title('Pendulum vs perturbation series')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.ylim(0, 12)
    plt.tight_layout()
    plt.savefig('pendulum_vs_series_0_to_2pi_RK4.pdf')
    plt.show()
    """


    """ 1.3
    mass = 1
    omega0 = 2
    c_harmonic = omega0**2
    x0, v0 = 1, 0
    gamma_list = [0.5, 3.0]
    time_step = 1e-3
    tmax = 30
    line_width = 1.8

    def run_case(gamma_value):
        osc = Oscillator(m=mass, c=c_harmonic, t0=0, theta0=x0, dtheta0=v0, gamma=gamma_value)
        sim = Simulation(osc)
        sim.run(Harmonic(), RK4Integrator(time_step), tmax=tmax)
        time_array = np.array(sim.obs.time)
        x_array = np.array(sim.obs.pos)
        v_array = np.array(sim.obs.vel)

        energy_array = 0.5*mass*(v_array**2) + 0.5*mass*(omega0**2)*(x_array**2)
        return time_array, x_array, v_array, energy_array

    # estimate tau
    def estimate_tau_linear_from_peaks(time_array, x_array):
        
        #A0 = first |x| peak after t=0; target = A0 / e.
        #Find the first subsequent peak below target and linearly interpolate
        #between the two peak points (t_{k-1}, A_{k-1}) and (t_k, A_k).
        #Returns: tau (float or None), target (A0/e), (t1, A1), (t2, A2)
        
        abs_x = np.abs(x_array)
        # local maxima of |x|
        peak_idx = np.where((abs_x[1:-1] > abs_x[:-2]) & (abs_x[1:-1] >= abs_x[2:]))[0] + 1
        # avoid the very first samples near t=0
        peak_idx = peak_idx[time_array[peak_idx] > 2*time_step]
        if peak_idx.size < 2:
            return None, None, None, None

        t_peaks = time_array[peak_idx]
        A_peaks = abs_x[peak_idx]

        A0 = A_peaks[0]
        target = A0 / np.e

        for k in range(1, len(A_peaks)):
            if A_peaks[k] <= target:
                t1, A1 = t_peaks[k-1], A_peaks[k-1]
                t2, A2 = t_peaks[k],   A_peaks[k]
                if A2 == A1:
                    return t2, target, (t1, A1), (t2, A2)
                tau_value = t1 + (target - A1) * (t2 - t1) / (A2 - A1)
                return float(tau_value), target, (t1, A1), (t2, A2)

        return None, target, (t_peaks[-2], A_peaks[-2]), (t_peaks[-1], A_peaks[-1])

    for gamma_value in gamma_list:
        t, x, v, E = run_case(gamma_value)

        # print tau
        tau_est, target_A, p1, p2 = estimate_tau_linear_from_peaks(t, x)
        if tau_est is not None:
            print(f"gamma = {gamma_value:.3f} -> tau ≈ {tau_est:.4f} s (A0/e ≈ {target_A:.4f})")
        else:
            print(f"gamma = {gamma_value:.3f} -> tau could not be estimated from peaks")

        fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
        axes[0].plot(t, x, lw=line_width)
        axes[1].plot(t, v, lw=line_width)
        axes[2].plot(t, E, lw=line_width)

        # --- NEW: visualize A0/e and tau on x(t) plot (minimal) ---
        if target_A is not None:
            axes[0].axhline(target_A, color='k', ls='--', lw=1.0, alpha=0.6)
        if tau_est is not None:
            axes[0].axvline(tau_est, color='k', ls='--', lw=1.0, alpha=0.6)

        axes[0].set_ylabel("x(t)")
        axes[1].set_ylabel("v(t)")
        axes[2].set_ylabel("E(t)")
        axes[2].set_xlabel("time [s]")

        fig.suptitle(f"Damped HO: omega0=2, gamma={gamma_value}, x0=1, v0=0, dt={time_step}")
        fig.tight_layout()
        plt.savefig(f"dho_xt_vt_Et_gamma{gamma_value}.pdf")
        plt.show()

    delta_gamma = 0.01
    gamma_start = 3
    gamma_end   = 5.5

    gamma_value = gamma_start
    gamma_critical = None

    while gamma_value <= gamma_end:
        t, x, v, E = run_case(gamma_value)
        if np.all(x[1:] > 0): 
            gamma_critical = gamma_value
            break
        gamma_value = round(gamma_value + delta_gamma, 10)

    print("gamma_critical:", gamma_critical)
    """



    mass = 1.0
    c_pendulum = 4.0            # sqrt(g/L)=2  ->  c=g/L=4 (matches the template's convention)
    gamma = 1.0
    theta0 = 0.5*np.pi
    dtheta0 = 0.0

    time_step = 1e-3            # small step for smooth curve
    tmax = 30.0
    line_width = 1.8

    # ----- simulate damped pendulum with your classes -----
    oscillator = Oscillator(m=mass, c=c_pendulum, t0=0.0, theta0=theta0, dtheta0=dtheta0, gamma=gamma)
    simulation = Simulation(oscillator)
    integrator = RK4Integrator(time_step)
    system = Pendulum()

    simulation.run(system, integrator, tmax=tmax)

    # ----- phase portrait data -----
    theta_series = np.array(simulation.obs.pos)
    dtheta_series = np.array(simulation.obs.vel)

    # (optional) wrap angle to (-pi, pi] for a cleaner portrait:
    # theta_series = np.arctan2(np.sin(theta_series), np.cos(theta_series))

    # ----- plot θ̇ vs θ -----
    plt.figure(figsize=(6,5))
    plt.plot(theta_series, dtheta_series, lw=line_width, label=r'trajectory')
    plt.plot(theta_series[0], dtheta_series[0], 'ro', ms=6, label='start')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\dot{\theta}$')
    plt.title(r'Damped pendulum phase portrait ( $\gamma=1$, $\sqrt{g/\ell}=2$ )')
    plt.grid(True, ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig('damped_pendulum_phase_portrait.pdf')
    plt.show()

    

if __name__ == "__main__" :
    exercise_11()
    # exercise_12()
    # ...
