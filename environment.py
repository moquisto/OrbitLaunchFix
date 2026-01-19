
import numpy as np
import ussa1976
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import config

class Atmosphere:
    """
    A class to provide atmospheric properties based on the US Standard Atmosphere 1976 model,
    extended to 1000 km. It uses a pre-computed lookup table and linear interpolation
    for high performance.


    WARNING: Numerical Discontinuity at max_alt

    The current implementation forces atmospheric density and pressure to strictly 0.0 for altitudes above 1000 km. While physically approximating a vacuum, this creates a sharp mathematical discontinuity (infinite gradient) at the boundary.

    This presents two critical risks for rocket simulations:

    Solver Stalling: Variable-step ODE solvers (e.g., scipy.integrate.solve_ivp, Runge-Kutta) may "stutter" or hang as they reduce the time step to near-zero in an attempt to resolve the non-smooth transition.

    Mathematical Singularities: Any guidance algorithm, entropy calculation, or plotting function that computes logarithmic values (e.g., np.log(rho)) will crash, as the logarithm of zero is undefined (ln(0)=−∞).

    Recommendation: For high-fidelity simulations, clamp the minimum return value to a non-zero epsilon (e.g., 1e-100) rather than 0.0.


    """
    def __init__(self, step=config.ATMOSPHERE_STEP, max_alt=config.ATMOSPHERE_MAX_ALT):
        """
        Initializes the atmospheric model by generating a pre-computed lookup table.

        Args:
            step (int): The altitude step size in meters for the lookup table.
            max_alt (int): The maximum altitude in meters for the lookup table.
        """
        self._step = step
        self._max_alt = max_alt
        self._grid = np.arange(0, self._max_alt + self._step, self._step)
        
        # Generate the atmospheric data using the ussa1976 library
        self._data = ussa1976.compute(z=self._grid)
        
        # Extract the values for interpolation
        altitudes = self._grid
        densities = self._data["rho"].values
        pressures = self._data["p"].values
        temperatures = self._data["t"].values
        
        # Create linear interpolation functions
        self._rho_interp = interp1d(altitudes, densities, kind='linear', fill_value="extrapolate")
        self._p_interp = interp1d(altitudes, pressures, kind='linear', fill_value="extrapolate")
        self._T_interp = interp1d(altitudes, temperatures, kind='linear', fill_value="extrapolate")

        # Store the temperature at the boundary for requests above max_alt
        self._temp_at_boundary = temperatures[-1]

    def get_properties(self, altitude):
        """
        Returns the atmospheric properties (density, pressure, temperature) for a given altitude.

        Args:
            altitude (float): The altitude in meters.

        Returns:
            tuple: A tuple containing three floats: (rho, p, T).
                   rho: Density in kg/m³
                   p: Pressure in Pa
                   T: Temperature in K
        """
        # Clamp altitude to be within [0, max_alt]
        if altitude < 0:
            altitude = 0
            
        if altitude > self._max_alt:
            return 0.0, 0.0, self._temp_at_boundary

        rho = self._rho_interp(altitude).item()
        p = self._p_interp(altitude).item()
        T = self._T_interp(altitude).item()
        
        return rho, p, T






if __name__ == '__main__':
    # Initialize the atmospheric model
    atmosphere_model = Atmosphere()

    # Generate a range of altitudes to test the interpolation
    test_altitudes = np.linspace(0, 1050000, 2000)
    
    # Get the properties for each altitude
    results = [atmosphere_model.get_properties(alt) for alt in test_altitudes]
    
    # Unzip the results
    rhos, ps, Ts = zip(*results)

    # Create the plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot Density vs. Altitude
    ax1.plot(test_altitudes / 1000, rhos)
    ax1.set_xlabel("Altitude (km)")
    ax1.set_ylabel("Density (kg/m³)")
    ax1.set_title("Density vs. Altitude")
    ax1.grid(True)
    ax1.set_yscale('log')

    # Plot Temperature vs. Altitude
    ax2.plot(test_altitudes / 1000, Ts)
    ax2.set_xlabel("Altitude (km)")
    ax2.set_ylabel("Temperature (K)")
    ax2.set_title("Temperature vs. Altitude")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()




