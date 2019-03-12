import numpy as np
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

class BaseStar():
    """
    Baseclass for star solvers. Must implement density function in subclasses. 
    Assumes scaled, geometrized units where G = c = M_sun = 1.
    
    TODO:
        - Fix scalar field so it matches minkowski metric
    """
    def __init__(self):
        self.initialized = False
        self._allow_negative_pressure = True
        
    
    def set_initial_conditions(self, rhoc, r_max_cgs = 50e5,  Nr = 2000,
                               length_cgs_to_scaled = 6.7706e-6):
        """
        rhoc : float
            central density in scaled, geometrized units.
        r_max : float
            maximum integration radius in km
        Nr : int
            number of integration points
            """
        self.rhoc = rhoc
        self.r_max = r_max_cgs * length_cgs_to_scaled # from cm to scaled units
        
        self.Nr = Nr
        self.initialized = True
        return self
    
    def rho0(self, P):
        raise NotImplementedError
        
    def rho(self, P, rho0 = None):
        raise NotImplementedError
        
    def P(self, rho):
        raise NotImplementedError
        
    def derivatives(self, r, y):
        (m, P, Phi, M0) = y

        if not self._allow_negative_pressure:
            if P < 10**self.pres_bounds[0] or np.isnan(P):
                return [0,0,0,0]

        rho = self.rho(P) 
        rho0 = self.rho0(P, rho) 
        
        dmdr = 4*np.pi*r**2*rho
        
        # Assuming m == 0 for r == 0
        # if m == 0:
        #     dPdr = 0
        #     dM0dr = 4*np.pi*r**2*rho0
        # else:
        dPdr = - rho*m/r**2*(1 + P/rho)*(1 + 4*np.pi*P*r**3/m)/(1-2*m/r)
        dM0dr = 4*np.pi*r**2*rho0/np.sqrt(1-2*m/r)

        dPhidr = - 1/rho * dPdr / ( 1 + P/rho )
        return [dmdr, dPdr, dPhidr, dM0dr]


    def get_pressure_event(self):
        """Passed to solver for termination when is pressure 0."""
        def pressure_event(t, y):
            # print(y)
            if np.isnan(y[1]):
                return -1
            else:
                return y[1] - 10**self.pres_bounds[0]
        pressure_event.terminal = True
        return pressure_event
        
    def solve_star_ivp(self, integrator = 'dopri5', tol = 1e-6):
        """Uses scipy.solve_ivp to solve the TOV-equations. Works for
        Polytrope stars. Has trouble finding the edge for table-stars, as
        table lookup for out-of-bounds pressures is hard to prevent."""

        if not self.initialized:
            raise ValueError('Must set initial conditions first!')

        Pc   = self.P(self.rhoc)
        rho0 = self.rho0(Pc)

        r0   = self.r_max * 1e-8 # small nonzero radius to start

        m    = 4/3*np.pi*r0**3 * self.rhoc
        M0   = 4/3*np.pi*r0**3 * rho0 / np.sqrt(1-2*m/r0)
        
        Phi  = 0
        y0   = [m, Pc, Phi, M0]

        solver = scipy.integrate.solve_ivp(self.derivatives, 
                                           t_span = [r0, self.r_max], y0 = y0,
                                           events = self.get_pressure_event())
        return solver

    def solve_star_ode(self, integrator = 'dopri5', tol = 1e-6, Nr = 100):
        """Uses scipy.ode to solve the TOV-equations, with controlled edge
        finding. 
        
        MIGHT BE DEPRECATED, if interpolate with extrapolation at edges
        works.
        """
        if not self.initialized:
            raise ValueError('Must set initial conditions first!')
        
        Pc = self.P(self.rhoc)
        rho0 = self.rho0(Pc)

        r0 = self.r_max * 1e-8 # small nonzero radius to start

        m = 4/3*np.pi*r0**3 * self.rhoc
        M0 = 4/3*np.pi*r0**3 * rho0 / np.sqrt(1-2*m/r0)
        
        Phi = 0
        y0 = [m, Pc, Phi, M0]


        solver = scipy.integrate.ode(self.derivatives).set_integrator(integrator)
        solver.set_initial_value(y0, r0)
        
        dr =  self.r_max / Nr

        t = [r0]
        y = [y0]
        
        P = Pc
        dr_tol = dr * 1e-4
        
        while solver.successful() and solver.t < self.r_max and (not P < 10**self.pres_bounds[0])  and (not np.isnan(P)):
            # print(m)
            solver.integrate(solver.t+dr)

            if solver.y[1] < 10**self.pres_bounds[0]:
                # overstepped and produced invalid P
                # go back and recalculate with smaller step until 
                # dr < dr_tol
                dr /= 2
                if dr < dr_tol:
                    break

                # reset solver to previous values
                solver.set_initial_value(y[-1], t[-1])
            elif m < 0:
                raise ValueError('negative mass!')
            else:
                y.append(solver.y)
                t.append(solver.t)

            m = solver.y[0]
            P = solver.y[1]
                
        if not solver.successful():
            import warnings
            warnings.warn('Something went wrong in the integration, and we need a better error message')
            # print("P", P, 'm', m)
        t = np.array(t)
        y = np.array(y)
        return solver, t, y
