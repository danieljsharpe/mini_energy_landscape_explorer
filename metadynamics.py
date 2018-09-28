'''
A Python program to perform a metadynamics simulation on a model potential energy surface to determine the
corresponding free energy surface
Daniel J. Sharpe
June 2018
'''

import numpy as np
import math
from model_potentials import Model_pes

class Metadynamics(object):

    k_B = 1.38064852*10.**(-23) # Boltzmann constant

    def __init__(self, x_0, v_0, dt, nsteps, omega, tau_f, sigma_b, pot_func, wt=False, delta_T=None):
        self.x = x_0 # coords set to initial coords
        self.v = v_0 # velocities set to initial values
        self.dt = dt # timestep for dynamics
        self.nsteps = nsteps # max no of MD steps
        self.omega = omega # bias deposition rate
        self.tau_f = tau_f # flooding time
        self.n_update = math.floor(self.tau_f/self.dt) # update interval (in no. of steps) for Gaussian
        self.sigma_b = sigma_b # standard deviation in Gaussian bias
        self.pot_func = pot_func # handle for potential function
        self.ndim = np.shape(self.x)[0] # dimensionality of potential (no. of 'collective variables' (CVs))
        self.sigma = np.array([sigma_b]*self.ndim) # initialise standard deviation of Gaussian bias potential
        self.mu = np.zeros(self.ndim) # initialise mean of Gaussian bias potential
        self.wt = wt # use tempering T/F
        if self.wt: # use well-tempered metadynamics (WT-MTD)
            self.delta_T = delta_T # temperature difference parameter for WT-MTD

    # function to update the Gaussian bias potential
    # note that the height of each deposited Gaussian is given by: tau_f*omega
    def update_bias_pot(self):
        for i in range(self.ndim):
            self.mu[i] += self.x[i]
            self.sigma[i] = math.sqrt(self.sigma[i]**2+self.sigma[i]**2)

    # function to rescale the height of the Gaussian bias potential (in WT-MTD)
    def scale_bias_height(self):
        self.W *= np.exp((-self.omega*self.tau_f*np.exp(-0.5*(self.xold-self.x)/self.sigma)**2) \
                         / (k_B*self.delta_T))

    def calc_bias_force(self):
        f_bias = 0.
        return f_bias

    def calc_md_force(self):
        f_md = 0.
        return f_md

    # A N-dimensional Gaussian function (not normalised), such as is deposited at regular intervals on the PES
    # mean value mu & standard deviation sigma
    @staticmethod
    def gaussian(x, mu, sigma):
        return np.exp(-0.5*((x-mu)/sigma)**2)

    # main function to propagate biased MD and explore the PES
    def propagate(self):
        i = 1
        while i < self.nsteps:
            print "iteration:", i, "coords:", self.x
            self.x_old = self.x
            if i != 0:# and int(i/self.n_update) == 1:
                self.update_bias_pot()
                if self.wt:
                    self.scale_bias_height()
            print "mu:", self.mu, "sigma:", self.sigma
            v_bias = self.omega*self.tau_f*Metadynamics.gaussian(self.x,self.mu,self.sigma)
            print v_bias
            f_md = self.calc_md_force()
            f_bias = self.calc_bias_force()
            self.x += 0.1
            i += 1
            if i == 3:
                term_1 = self.omega*self.tau_f*Metadynamics.gaussian(self.x,np.array([0.3,0.35]),self.sigma)
                term_2 = self.omega*self.tau_f*Metadynamics.gaussian(self.x,np.array([0.4,0.45]),self.sigma)
                print term_1+term_2
                break


if __name__ == "__main__":
    '''
    # EXAMPLE 1
    pes = Model_pes("Leps-HO")
    pot_func = pes.leps_ho_pot
    mtd1 = Metadynamics(
    '''
    # EXAMPLE 2
    pes2 = Model_pes("I shouldnt have to give a string")
    pot_func = pes2.three_hole_pot
    mtd2 = Metadynamics(np.array([0.3,0.35]),[0.1,-0.1],0.01,500,0.35,0.2,0.1,pot_func)
    mtd2.propagate()
