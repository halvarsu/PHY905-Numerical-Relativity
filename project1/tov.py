#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def deriv_P(r, rho, m, P):
    return - rho * m / r**2 * (1 +  P /rho) * ( 1 + 4*np.pi*P*r**3/m)/(1-2*m/r)

def deriv_m(r, rho):
    return 4*np.pi*r**2 * rho

def deriv_Phi(r, rho, P, dPdr):
    return - 1/rho * dPdr / ( 1 + P/rho )

def f(r, y):
    (m, P, Phi) = y
    rho0 = P ** (1/gamma)
    
    rho = rho0 + P/(gamma - 1)
    dmdr = 4 * np.pi * r**2 * rho
    if r == 0 or m == 0:
        dPdr = 0
    else:
        dPdr = - rho * m / r**2 * ((1 +  P /rho)                                  * ( 1 + 4*np.pi*P*r**3/m)/(1-2*m/r))
    dPhidr = - 1/rho * dPdr / ( 1 + P/rho )
    if verbose:
        print(0, r, m, rho)
        print(1, dmdr)
        print(2, dPdr)
        print(3, dPhidr)
    return [dmdr, dPdr, dPhidr]

class BaseStar():
    """
    Baseclass for star solvers. Must implement density function in subclasses. 
    Assumes scaled, geometrized units where G = c = M_sun = 1.
    
    TODO:
        - Fix scalar field so it matches minkowski metric
        - add integration method for rest-mass M_0
    """
    def __init__(self):
        self.initialized = False
        self.pres_bounds = [0, np.inf]
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

        rho0 = self.rho0(P) 
        rho = self.rho(P) 
            # print("HALLO", rho0, rho)
        
        dmdr = 4*np.pi*r**2*rho
        if m == 0:
            dPdr = 0
            dM0dr = 4*np.pi*r**2*rho0
        else:
            dPdr = - rho*m/r**2*(1 + P/rho)*(1 + 4*np.pi*P*r**3/m)/(1-2*m/r)
            dM0dr = 4*np.pi*r**2*rho0/np.sqrt(1-2*m/r)
        dPhidr = - 1/rho * dPdr / ( 1 + P/rho )
        # print([dmdr, dPdr, dPhidr, dM0dr])
        # print(P, rho)
        return [dmdr, dPdr, dPhidr, dM0dr]


    def get_pressure_event(self):
        def pressure_event(t,y):
            """Passed to solver for termination when is pressure 0."""
            return y[1] 
        pressure_event.terminal = True
        return pressure_event
        
    def solve_star_ivp(self, integrator = 'dopri5', tol = 1e-6):
        """Uses scipy.solve_ivp to solve the TOV-equations. Has trouble
        finding the edge for table-stars, as table lookup for out-of-bounds
        pressures is hard to stop."""
        Pc = self.P(self.rhoc)
        rho0 = self.rho0(Pc)

        r0 = self.r_max * 1e-8 # small nonzero radius to start

        m = 4/3*np.pi*r0**3 * self.rhoc
        M0 = 4/3*np.pi*r0**3 * rho0 / np.sqrt(1-2*m/r0)
        
        Phi = 0
        y0 = [m, Pc, Phi, M0]

        solver = solve_ivp(self.derivatives, t_span = [r0, self.r_max],  y0 = y0,
                          events = self.get_pressure_event())
        return solver

    def solve_star_ode(self, integrator = 'dopri5', tol = 1e-6, Nr = 100):
        """Uses scipy.ode to solve the TOV-equations, with controlled edge
        finding."""
        if not self.initialized:
            raise ValueError('Must set initial conditions first!')
        
        Pc = self.P(self.rhoc)
        rho0 = self.rho0(Pc)

        r0 = self.r_max * 1e-8 # small nonzero radius to start

        m = 4/3*np.pi*r0**3 * self.rhoc
        M0 = 4/3*np.pi*r0**3 * rho0 / np.sqrt(1-2*m/r0)
        
        Phi = 0
        y0 = [m, Pc, Phi, M0]


        from scipy.integrate import ode
        solver = ode(self.derivatives).set_integrator(integrator)
        solver.set_initial_value(y0, r0)
        
        dr =  self.r_max / Nr

        t = []
        y = []
        
        m = 0
        P = Pc
        dr_tol = dr *1e-6
        
        while solver.successful() and solver.t < self.r_max and (not P < 10**self.pres_bounds[0])  and (not np.isnan(P)):
            # print(1, P, P < 0)
            solver.integrate(solver.t+dr)
            # print(2, solver.y[1], solver.y[1] < 0)

            y.append(solver.y)
            t.append(solver.t)
            
            if P == solver.y[1] or solver.y[1] < 10**self.pres_bounds[0]:
                # print(P)
                # overstepped and produced negative P
                # go back and recalculate with smaller step
                dr /= 2
                if dr < dr_tol:
                    break
                #print(dr)
                ## Remove values and reset solver to previous 
                #print(P, t[-3:])
                y.pop()
                t.pop()
                solver.set_initial_value(y[-1], t[-1])
                # print(3333, y[-1], solver.y)
                m = solver.y[0]
                P = solver.y[1]
                continue
            m = solver.y[0]
            P = solver.y[1]
            if m < 0:
                raise ValueError('negative mass!')
                
        if not solver.successful():
            import warnings
            warnings.warn('Something went wrong in the integration, and we need a better error message')
        y = np.array(y)
        return solver, t, y
    

class PolytropeStar(BaseStar):
    def __init__(self, gamma, K = 1, *args, **kwargs):
        self.K = K
        self._gamma = gamma
        self._n = 1/(gamma - 1)
        self.pres_bounds = [-np.inf, np.inf]
        self.pres_bounds = [-np.inf, np.inf]
        self.pres_bounds = [-np.inf, np.inf]

        BaseStar.__init__(self, *args, **kwargs)
        
    def rho0(self, P):
        return (P/self.K) ** (1/self._gamma)
    
    def rho(self, P, rho0 = None):
        rho0 = rho0 or self.rho0(P)
        return rho0 + P/(self._gamma - 1)
    
    def P(self, rho):
        return self.K*rho**self._gamma 
