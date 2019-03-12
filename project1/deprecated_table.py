import numpy as np
import h5py
from tov import BaseStar
from scipy.interpolate import (InterpolatedUnivariateSpline, 
        interp1d, LinearNDInterpolator)


class TableStar(BaseStar):
    def __init__(self, tablefile, scaling = {}, *args, **kwargs):
        """
        1st attempt.
        Uses 2D interpolation to find minmal munu
        
        Assumes cold star, i.e minimal T. 
        
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
