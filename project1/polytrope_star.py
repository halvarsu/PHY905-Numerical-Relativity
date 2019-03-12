#!/usr/bin/env python
# coding: utf-8

import scipy
import numpy as np
import matplotlib.pyplot as plt
from base_star import BaseStar
from scipy.integrate import solve_ivp, ode
from scipy.interpolate import LinearNDInterpolator


class PolytropeStar(BaseStar):
        """Implements pressure and density methods for a polytropic star."""
    def __init__(self, gamma, K = 1, *args, **kwargs):
        self.K = K
        self._gamma = gamma
        self._n = 1/(gamma - 1)
        self.pres_bounds = [-np.inf, np.inf] # in log space

        BaseStar.__init__(self, *args, **kwargs)
        
    def rho0(self, P):
        """Finds rho0 from P. """
        return (P/self.K) ** (1/self._gamma)
    
    def rho(self, P = None, rho0 = None):
        """Can find rho from either P or rho0"""
        if not rho0 is None:
            rho0eps = self.K*rho0**self._gamma / (self._gamma - 1)
        elif not P is None:
            rho0 = self.rho0(P)
            rho0eps = P/(self._gamma -1)
        else:
            raise ValueError('Either P or rho0 must be specified')
        return rho0 + rho0eps
    
    def P(self, rho):
        return (self.K*rho)**self._gamma 
