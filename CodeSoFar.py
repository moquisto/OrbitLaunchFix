import casadi as ca
import numpy as np
# AI check for notes and stuff and some cleaning of the code I guess,
#mostly on the comments, a lot of things to add and improve. Especially the dynamics.
# NOTE: Good choice on Direct Transcription.
# It is generally more robust for this than Shooting methods.

# test values:
m_init = 10000
m_dry = 5000
target_height = 200000  # 200 kilometers

class RocketTrajectoryOptimizer:
    def __init__(self, target_h, m_initial, m_dry):
        """
        Initialize the optimization environment, define constants, 
        and determine node count based on target height.
        """
        self.target_h = target_h
        self.m_initial = m_initial
        self.m_dry = m_dry

        self.opti = ca.Opti()
        self.n_nodes = self._calculate_nodes(target_h)
        # TODO: You defined the class but haven't called the setup methods yet.
        # usually, you don't call them in __init__ to keep it clean, 
        # but ensure you call them in the right order in the "Usage Flow" at the bottom.
        pass

    def _calculate_nodes(self, h):
        """
        Logic to return the number of nodes (N) as a function of 
        the orbit height h to ensure simulation resolution.
        """
        n_nodes = h // 1000  
        return n_nodes

    def define_states(self):
        """
        Define the decision variables for the state vector:
        X = [px, py, pz, vx, vy, vz, mass] at each node.
        """
        self.X = self.opti.variable(7, self.n_nodes + 1)
        
        self.pos = self.X[0:3, :]
        self.vel = self.X[3:6, :]
        self.mass = self.X[6, :]
        pass

    def define_controls(self):
        """
        Define the decision variables for the guidance:
        U = [tx, ty, tz] (Thrust vector in 3D) at each node.
        """
        self.U = self.opti.variable(3, self.n_nodes)
        
        self.thrust_x = self.U[0, :]
        self.thrust_y = self.U[1, :]
        self.thrust_z = self.U[2, :]

        # FIX: "Magic Number" 10000. 
        # Define max_thrust as a class variable (self.max_thrust) in __init__ 
        # so you can change it easily later.
        
        # OPTIMIZATION: Python loops are slow in CasADi construction.
        # Consider vectorizing this:
        # thrust_sq = ca.sum1(self.U**2)
        # self.opti.subject_to(thrust_sq <= self.max_thrust**2)
        for k in range(self.n_nodes):
            thrust_magnitude = ca.norm_2(self.U[:, k])
            self.opti.subject_to(thrust_magnitude <= 10000) 
        pass

    def get_dynamics(self): 
        # FIX: You previously had 'def get_dynamics(self, x, u):'
        # But you don't use those arguments x, u inside; you create new MX symbols.
        # Remove x, u from arguments to avoid confusion.
        
        """
        Construct casadi function for F = ma
        """
        px, py, pz = ca.MX.sym("px"), ca.MX.sym("py"), ca.MX.sym("pz")
        vx, vy, vz = ca.MX.sym("vx"), ca.MX.sym("vy"), ca.MX.sym("vz")
        m = ca.MX.sym("m")
        tx, ty, tz = ca.MX.sym("tx"), ca.MX.sym("ty"), ca.MX.sym("tz")

        x = ca.vertcat(px, py, pz, vx, vy, vz, m)
        u = ca.vertcat(tx, ty, tz)

        # NOTE: Consistency Check.
        # In apply_boundary_conditions you used R=6731000. Here you use 6371000.
        # Move these constants to __init__ (self.Re, self.mu, etc.) to ensure they match everywhere.
        mu = 3.986e14       
        Re = 6371000        
        g0 = 9.81           
        Isp = 300.0         
        CD = 0.5            
        A_ref = 10.0        
        rho0 = 1.225        
        H_scale = 7200.0    

        # Calculations
        pos = x[0:3]
        vel = x[3:6]
        # mass = x[6] # Unused variable assignment
        
        pos_magnitude = ca.norm_2(x[0:3])
        vel_magnitude = ca.norm_2(x[3:6])
        altitude = pos_magnitude - Re

        gravitational_acceleration = (-mu / pos_magnitude**3) * pos

        rho = rho0 * ca.exp(-altitude / H_scale)
        
        # NOTE: Drag Logic looks correct (opposing velocity).
        drag_force = -0.5 * rho * vel_magnitude * CD * A_ref * vel

        thrust_force = u
        thrust_magnitude = ca.norm_2(thrust_force)

        # Newton's Second Law
        tot_acceleration = gravitational_acceleration + (thrust_force + drag_force) / m

        # Mass flow
        mass_flow = -thrust_magnitude / (Isp * g0)

        x_dot = ca.vertcat(vx, vy, vz, tot_acceleration[0], tot_acceleration[1], tot_acceleration[2], mass_flow)
        
        # This creates the function self.f that you will call later
        self.f = ca.Function("f", [x, u], [x_dot])

    def apply_physics_constraints(self):
        """
        Apply Runge-Kutta 4 (RK4) integration constraints.
        """
        # TODO: Check that self.f exists.
        # You must call self.get_dynamics() BEFORE calling this function.

        self.T = self.opti.variable()
        self.opti.subject_to(self.T >= 0)
        self.opti.set_initial(self.T, 300) 

        dt = self.T / self.n_nodes

        for k in range(self.n_nodes):
            x_k = self.X[:, k]
            u_k = self.U[:, k]
            x_k_next = self.X[:, k + 1]

            # RK4 Step
            k1 = self.f(x_k, u_k)
            k2 = self.f(x_k + dt/2 * k1, u_k) # FIX: CasADi might complain about mixed types here.
                                              # If it fails, ensure dt is treated as a scalar MX.
            k3 = self.f(x_k + dt/2 * k2, u_k)
            k4 = self.f(x_k + dt * k3, u_k)

            x_k_next_rk4 = x_k + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

            self.opti.subject_to(x_k_next == x_k_next_rk4)

    def apply_boundary_conditions(self):
        # FIX: Inconsistent Earth Radius (see get_dynamics note).
        R_earth = 6731000
        
        # Start Conditions
        self.opti.subject_to(self.pos[0, 0] == R_earth)
        self.opti.subject_to(self.pos[1, 0] == 0)
        self.opti.subject_to(self.pos[2, 0] == 0)
        self.opti.subject_to(self.vel[:, 0] == 0)
        self.opti.subject_to(self.mass[0] == self.m_initial)

        # End Conditions
        r_target_square = (R_earth + self.target_h)**2
        r_final_square = ca.sumsqr(self.pos[:, -1])
        self.opti.subject_to(r_final_square >= r_target_square)

        # TODO: MISSING ORBITAL VELOCITY CONSTRAINT
        # Right now, the rocket can just reach 200km with 0 velocity (fall back down).
        # You need to enforce tangential velocity:
        # v_orbit = sqrt(mu / r_target)
        # self.opti.subject_to(ca.sumsqr(self.vel[:, -1]) >= v_orbit**2)
        # AND you should enforce that radial velocity is 0 (circular orbit).
        # dot_product(pos_final, vel_final) == 0

    def apply_path_constraints(self):
        R_earth = 6731000
        
        # FIX: LOGIC ERROR
        # You wrote: self.mass[:] <= self.m_dry
        # This says "Mass must be LESS than empty tank".
        # It should be: self.mass[:] >= self.m_dry
        self.opti.subject_to(self.mass[:] <= self.m_dry) 

        max_angle = 10 
        boundary_cos_square = np.cos(np.radians(max_angle))**2
        
        for k in range(1, self.n_nodes):
            dot_prod = ca.dot(self.U[:, k], self.vel[:, k])
            u_k = ca.dot(self.U[:, k], self.U[:, k])
            v_k = ca.dot(self.vel[:, k], self.vel[:, k])
            
            # NOTE: Good use of the dot_prod >= 0 check!
            self.opti.subject_to(dot_prod >= 0)
            self.opti.subject_to(dot_prod**2 >= boundary_cos_square * u_k * v_k) # FIX: Check inequality sign.
            # If (u.v)^2 is the square of the cosine...
            # if angles are close, cosine is close to 1.
            # if angles are far, cosine is small.
            # You want cos^2 >= limit. (You had <= in your code, which forces angle to be LARGE).
            # PLEASE DOUBLE CHECK THIS LOGIC.

        pos_sq = self.pos**2
        square_magnitude = ca.sum1(pos_sq)
        self.opti.subject_to(square_magnitude >= R_earth**2)

    def set_objective(self):
        self.opti.minimize(-self.mass[-1])
        pass

# FIX: INDENTATION ERROR -----------------------------------------------------
# The following two functions are NOT part of the class. 
# They are indented at level 0. They need to be indented to match the class methods.

def set_initial_guess(self):
    # ... (Code omitted for brevity, but needs indenting)
    pass

def solve(self):
    # ... (Code omitted for brevity, but needs indenting)
    pass
# ----------------------------------------------------------------------------

# --- USAGE FLOW CORRECTIONS ---
sim = RocketTrajectoryOptimizer(target_h=target_height, m_initial=m_init, m_dry=m_dry)

sim.define_states()
sim.define_controls()

# TODO: CALL get_dynamics() HERE
# You must build the function 'f' before apply_physics_constraints tries to use it.
sim.get_dynamics() 

sim.apply_physics_constraints()
sim.apply_boundary_conditions()
sim.apply_path_constraints()
sim.set_objective()

# TODO: CALL sim.set_initial_guess() (Once indented properly)

# solution = sim.solve() (Once indented properly)
