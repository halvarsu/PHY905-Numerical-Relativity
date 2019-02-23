#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

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
    def __init__(self, gamma, K = 1):
        self.K = K
        self._gamma = gamma
        self._n = 1/(gamma - 1)
        self.initialized = False
        
    
    def set_initial_conditions(self, rhoc, r_max_cgs = 20e5,  Nr = 2000,
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
        
        rho0 = self.rho0(P) 
        rho = self.rho(P, rho0) 
        
        dmdr = 4*np.pi*r**2*rho
        if m == 0:
            dPdr = 0
            dM0dr = 4*np.pi*r**2*rho0
        else:
            dPdr = - rho*m/r**2*(1 + P/rho)*(1 + 4*np.pi*P*r**3/m)/(1-2*m/r)
            dM0dr = 4*np.pi*r**2*rho0/np.sqrt(1-2*m/r)
        dPhidr = - 1/rho * dPdr / ( 1 + P/rho )
        return [dmdr, dPdr, dPhidr, dM0dr]
    
        
    def solve_star(self, integrator = 'dopri5', tol = 1e-6):
        if not self.initialized:
            raise ValueError('Must set initial conditions first!')
        rhoc = self.rhoc
        Pc = self.P(rhoc)
        rho0 = self.rho0(Pc) # rhoc - Pc/(self._gamma - 1)

        solver = ode(self.derivatives).set_integrator(integrator)

        m = M0 = Phi = 0
        y0 = [m, Pc, Phi, M0]
        solver.set_initial_value(y0, 0)
        
        dr =  self.r_max / self.Nr

        values = []
        r_values = []
        
        m = 0
        P = Pc
        
        while solver.successful() and solver.t < self.r_max or self.P < 0:
            # if P < 0:     
            #     # overstepped, go back and recalculate with smaller step
            #     dr /= 2
            #     values.pop()
            #     r_values.pop()
            # elif P < tol: # break out of loop
            #     break

            solver.integrate(solver.t+dr)

            values.append(solver.y)
            r_values.append(solver.t)
            
            m = solver.y[0]
            P = solver.y[1]
            if m < 0:
                raise ValueError('negative mass!')
                
        if not solver.successful():
            import warnings
            warnings.warn('Something went wrong in the integration, and we need a better error message')

        values = np.array(values)
        return solver, values, r_values

    def maximum_mass(self, n):
        pass
    
class PolytropeStar(BaseStar):
    def __init__(self, *args, **kwargs):
        BaseStar.__init__(self, *args, **kwargs)
        
    def rho0(self, P):
        return (P/self.K) ** (1/self._gamma)
    
    def rho(self, P, rho0 = None):
        rho0 = rho0 or self.rho0(P)
        return rho0 + P/(self._gamma - 1)
    
    def P(self, rho):
        return self.K*rho**self._gamma 

