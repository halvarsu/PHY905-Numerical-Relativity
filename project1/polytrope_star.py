#!/usr/bin/env python
# coding: utf-8

import scipy
import numpy as np
import matplotlib.pyplot as plt
from base_star import BaseStar
from scipy.integrate import solve_ivp, ode
from scipy.interpolate import LinearNDInterpolator


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
