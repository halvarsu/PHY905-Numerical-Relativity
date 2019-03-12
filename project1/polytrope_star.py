#!/usr/bin/env python
# coding: utf-8


import h5py
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, ode
from scipy.interpolate import LinearNDInterpolator


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
    

class PolytropeStar(BaseStar):
    def __init__(self, gamma, K = 1, *args, **kwargs):
        self.K = K
        self._gamma = gamma
        self._n = 1/(gamma - 1)
        self.pres_bounds = [0, np.inf]

        BaseStar.__init__(self, *args, **kwargs)
        
    def rho0(self, P):
        return (P/self.K) ** (1/self._gamma)
    
    def rho(self, P, rho0 = None):
        rho0 = rho0 or self.rho0(P)
        return rho0 + P/(self._gamma - 1)
    
    def P(self, rho):
        return self.K*rho**self._gamma 



class TableStar(BaseStar):
    def __init__(self, tablefile, scaling = {}, *args, **kwargs):
        """Assumes cold star, i.e minimal T. 
        
        Scales to geometrized units c = G = M_sun = 1
        
        """
        BaseStar.__init__(self, *args, **kwargs)
        self._allow_negative_pressure = False

        f = h5py.File(tablefile, 'r')
        
        self.scaling = {}
        
        # default scaling is  c = G = M_sun = 1
        default_scaling = {'pres'  : 1.801569643420104e-39, # from dyn/cm^2
                           'rho'   : 1.6191700468788605e-18, # from g/cm^3
                           'specific_energy': 1.11265005605e-21,  # from erg/g
                           # munu should be MeV/baryon
                          }
        default_scaling.update(scaling)
        self.scaling.update(default_scaling)
        
        self.ye_arr   = f['ye'][:] 
        self.rho_arr  = f['logrho'][:] + np.log10(self.scaling['rho'])
        
        self.energy_shift  = f['energy_shift'][0] * self.scaling['specific_energy']

        rho_grid, ye_grid  = np.meshgrid(self.rho_arr, self.ye_arr)
        pres_values   = f['logpress'][:,0,:]  + np.log10(self.scaling['pres'])
        munu_values   = f['munu'][:,0,:]      # *          self.scaling['energy']
        energy_values = f['logenergy'][:,0,:] + np.log10(self.scaling['specific_energy']) 

        f.close()
        self.ye_bounds   = (np.min(self.ye_arr),  np.max(self.ye_arr))
        self.rho_bounds  = (np.min(self.rho_arr), np.max(self.rho_arr))
        self.pres_bounds = (np.min(pres_values),  np.max(pres_values))


        ye_pres_points = np.column_stack([ye_grid.ravel(), pres_values.ravel()]),
        ye_rho_points  = np.column_stack([ye_grid.ravel(), rho_grid.ravel()]),
        
            
        self.rho_interp       = LinearNDInterpolator(ye_pres_points, rho_grid.ravel() )
        self.pres_interp      = LinearNDInterpolator(ye_rho_points, pres_values.ravel())
        self.energy_interp    = LinearNDInterpolator(ye_rho_points, energy_values.ravel())
        self.munu_interp_rho  = LinearNDInterpolator(ye_rho_points, munu_values.ravel())
        self.munu_interp_pres = LinearNDInterpolator(ye_pres_points, munu_values.ravel())
        
        # self.rho_interp_lin_scaled = lambda points: rho_scale*10**self.rho_interp(points)
        
    def check_in_bounds(self, val, val_type):
        if val_type == 'rho':
            bounds = self.rho_bounds
        elif val_type == 'pres':
            bounds = self.pres_bounds
        elif val_type == 'ye':
            bounds = self.ye_bounds
        else:
            raise ValueError('Invalid val_type, %s' %val_type)
        if not ((val <= bounds[1]) and (val >= bounds[0])):
            raise ValueError("Value '%s' out of bounds. %.5f not in interval [%.5f, %.5f]" % (val_type, val, bounds[0], bounds[1]))
        
        
    def energy(self, ye, rho):
        logrho = np.log10(rho)
        return 10**self.energy_interp(ye, logrho) - self.energy_shift
    
        
    def munu(self, ye, val, val_type):
        """Returns value of munu either from pressure or from density, based on val_type argument"""
        self.check_in_bounds(val, val_type)
        if val_type == 'rho':
            return self.munu_interp_rho([ye, val])[0]
        elif val_type == 'pres':
            return self.munu_interp_pres([ye, val])[0]
        else:
            raise ValueError("val_type must be either 'rho' or 'pres'")
        
    def P(self, rho):
        a, b = self.ye_bounds
        logrho = np.log10(rho)
        res = scipy.optimize.bisect(self.munu, a = a, b = b, args=(logrho, 'rho'))
        self.current_ye = res
        return 10**self.pres_interp(self.current_ye, logrho)
    
    def rho(self, P):
        a, b = self.ye_bounds
        logP = np.log10(P)
        res = scipy.optimize.bisect(self.munu, a = a, b = b,  args=(logP, 'pres'))
        self.current_ye = res
        return 10**self.rho_interp(self.current_ye, logP)
    
    def rho0(self, P, rho = None):
        """Uses P as is, no convert to log"""
        rho = rho or self.rho(P)
        eps = self.energy(self.current_ye, rho)
        return rho / (1 + eps)
    
