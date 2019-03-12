import numpy as np
import h5py
from base_star import BaseStar
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d


class BaseTableStar(BaseStar):
    def __init__(self, tablefile, scaling = {}, premade_tables = True, *args, **kwargs):
        """Star loading central values"""
        BaseStar.__init__(self, *args, **kwargs)
        self._allow_negative_pressure = True
        
        self.set_scaling(scaling)
        self.generate_interpolators(tablefile)
        
    def set_scaling(self, scaling = {}):
        """Sets scaling, default is from cgs to c=G=Msun=1"""
        # 
        default_scaling = {'pres': 1.801569643420104e-39, # from dyn/cm^2
                    'rho': 1.6191700468788605e-18, # from g/cm^3
                    'eps': 1.11265005605e-21,  # from erg/g 
                  }
        default_scaling.update(scaling)
        self.scaling = default_scaling
            
    def generate_interpolators(self, tablefile):
        raise NotImplementedError('Please use derived classes like TableStar or StevesStar')
            
    def energy(self, rho0):
        logrho0 = np.log10(rho0)
        return 10**self.eps_interp(logrho0) - self.energy_shift
    
    def P(self, rho0):
        logrho0 = np.log10(rho0)
        return 10**self.pres_interp(logrho0)
    
    def rho0(self, P):
        logP = np.log10(P)
        return 10**self.rho_interp(logP)
    
    def rho(self, P = None, rho0 = None):
        """Can find rho from rho0, or P by using P to find rho0."""
        
        if rho0 is None and P is None:
            raise ValueError('Either P or rho0 must be specified')
        
        rho0 or self.rho0(P)
        eps = self.energy(rho0)
        return rho0 / (1 - eps)


class TableStar(BaseTableStar):
    """Generates the interpolators from the full hdf5 file.
    
        TODO: Tidy up a bit more 
    """
    def generate_interpolators(self, tablefile):
        """Generates the interpolators from the full hdf5 file.
        """
        f = h5py.File(tablefile, 'r')
        self.ye_grid   = f['ye'][:] 
        self.rho_grid  = f['logrho'][:] + np.log10(self.scaling['rho'])
        
        self.energy_shift  = f['energy_shift'][0] * self.scaling['eps']

        rho_grid, ye_grid  = np.meshgrid(self.rho_grid, self.ye_grid)
        pres_values   = f['logpress'][:,0,:]  + np.log10(self.scaling['pres'])
        munu_values   = f['munu'][:,0,:]      # *          self.scaling['energy']
        energy_values = f['logenergy'][:,0,:] + np.log10(self.scaling['eps']) 
        f.close()

        self.ye_bounds   = (np.min(self.ye_grid),  np.max(self.ye_grid))
        self.rho_bounds  = (np.min(self.rho_grid), np.max(self.rho_grid))
        self.pres_bounds = (np.min(pres_values),  np.max(pres_values))

        self.energy_arr = np.zeros_like(self.rho_grid)
        self.pres_arr   = np.zeros_like(self.rho_grid)
        self.ye_arr     = np.zeros_like(self.rho_grid)

        prev_root = 0 
        
        # for each rho, extract the values given ye such that munu = 0
        for i, rho in enumerate(self.rho_grid):
            munu_func   = InterpolatedUnivariateSpline(self.ye_grid, munu_values[:,i])
            pres_func   = interp1d(self.ye_grid, pres_values[:,i])
            energy_func = interp1d(self.ye_grid, energy_values[:,i])
            
            ye_roots = munu_func.roots()
            
            
            if len(ye_roots) > 1:
                # in case several roots, we pick the closest one
                import warnings
                warnings.warn('Error, more than one root in munu! Using closest to previous')
                ye_root = ye_roots[np.argmin(np.abs(np.array(ye_roots) - prev_root))]
            elif len(ye_roots) == 0:
                # in case of no roots, we chose the previous value
                ye_root = prev_root
            else:
                ye_root = ye_roots[0]
                
            prev_root = ye_root
            
            self.ye_arr[i]      = ye_root
            self.pres_arr[i]    = pres_func(ye_root)
            self.energy_arr[i]  = energy_func(ye_root)
            
        # Generate the interpolators
        self.rho_interp    = interp1d(self.pres_arr, self.rho_grid, fill_value='extrapolate')
        self.eps_interp = interp1d(self.rho_grid, self.energy_arr, fill_value='extrapolate')
        self.pres_interp   = interp1d(self.rho_grid, self.pres_arr, fill_value='extrapolate')
        self.ye_interp = interp1d(self.rho_grid, self.ye_arr, fill_value='extrapolate')
        




class StevesStar(BaseTableStar):
    """Loads steves tables and generates interpolators, credits to Steve
    https://github.com/frommste"""
    def generate_interpolators(self, tablefile):
        """Loads pregenerated tablefile, i.e. steves files."""
        data = np.loadtxt(tablefile).T

        # first line contains energy shift
        with open(tablefile) as infile:
            line = infile.readline()
            self.energy_shift = float(line.split()[4]) * self.scaling['eps']

        logpres = data[0] + np.log10(self.scaling['pres'])
        logrho  = data[1] + np.log10(self.scaling['rho'])
        logeps  = data[2] + np.log10(self.scaling['eps']) 
        ye      = data[-1]

        self.ye_bounds   = (np.min(ye),  np.max(ye))
        self.rho_bounds  = (np.min(logrho), np.max(logrho))
        self.pres_bounds = (np.min(logpres), np.max(logpres))

        self.eps_interp  = interp1d(logrho, logeps, fill_value ='extrapolate')
        self.rho_interp  = interp1d(logpres, logrho, fill_value ='extrapolate')
        self.pres_interp = interp1d(logrho, logpres, fill_value ='extrapolate')
