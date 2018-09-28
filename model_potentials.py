'''
Python class for implementing various analytical model PESs
'''

import numpy as np

class Model_pes(object):

    # contains info on potential parameters
    def __init__(self, potential):

        if potential == 'Muller-Brown':
            self.AA = (-200., -100., -170., 15.)
            self.b = (0., 0., 11., 0.6)
            self.a = (-1., -1., -6.5, 0.7)
            self.c = (-10., -10., -6.5, 0.7)
            self.x0 = (1., 0., -0.5, -1.)
            self.y0 = (0., 0.5, 1.5, 1.)
        elif potential == 'Leps' or potential == 'Leps-HO':
                self.a = 0.05
                self.b = 0.30
                self.c = 0.05
                self.d_ab = 4.746
                self.d_bc = 4.746
                self.d_ac = 3.445
                self.r0_ab = 0.742
                self.r0_bc = 0.742
                self.r0_ac = 0.742
                self.alpha_ab = 1.942
                self.alpha_bc = 1.942
                self.alpha_ac = 1.942
                if potential == 'Leps-HO':
                    self.r_ac = 3.742
                    self.k_c = 0.2025 # force constant for harmonic coupling
                    self.c = 1.154
        return

    # Muller-Brown potential function. Takes 2D list of args. Warning - exhibits chaotic dynamics!
    # Global minimum at (-0.563, 1.417), other local minima at (-0.05, 0.47)  and (0.618, 0.01)
    # Transition states at (-0.822, 0.624) and (0.212, 0.293)
    # TP near (-2.8, 0)
    def mb_pot(self, x):
        potvalue = 0.0
        for i in range(4):
            potterm = self.AA[i]*(np.exp(self.a[i]*(x[0]-self.x0[i])**2 + \
                      self.b[i]*(x[0]-self.x0[i])*(x[1]-self.y0[i]) + \
                      self.c[i]*(x[1]-self.y0[i])**2))
            potvalue += potterm
        return potvalue

    # Classical Coulomb electrostatic function called by LEPS potential
    def coul(self, r, pair='ab'):
        if pair == 'ab':
            q = ((self.d_ab/2.0)*((3.0/2.0)*np.exp(-2.0*self.alpha_ab*(r-self.r0_ab)) - np.exp(-self.alpha_ab*(r-self.r0_ab)))) \
                / (1.0 + self.a)
        elif pair == 'bc':
            q = ((self.d_bc/2.0)*((3.0/2.0)*np.exp(-2.0*self.alpha_bc*(r-self.r0_bc)) - np.exp(-self.alpha_bc*(r-self.r0_bc)))) \
                / (1.0 + self.b)
        elif pair == 'ac':
            q = ((self.d_ac/2.0)*((3.0/2.0)*np.exp(-2.0*self.alpha_ac*(r-self.r0_ac)) - np.exp(-self.alpha_ac*(r-self.r0_ac)))) \
                / (1.0 + self.c)
        return q

    # Quantum exchange interaction function called by LEPS potential
    def exchange(self, r, pair='ab'):
        if pair == 'ab':
            j = (self.d_ab/4.0)*(np.exp(-2.0*self.alpha_ab*(r-self.r0_ab)) - 6.0*np.exp(-self.alpha_ab*(r-self.r0_ab)))
        elif pair == 'bc':
            j = (self.d_bc/4.0)*(np.exp(-2.0*self.alpha_bc*(r-self.r0_bc)) - 6.0*np.exp(-self.alpha_bc*(r-self.r0_bc)))
        elif pair == 'ac':
            j = (self.d_ac/4.0)*(np.exp(-2.0*self.alpha_ac*(r-self.r0_ac)) - 6.0*np.exp(-self.alpha_ac*(r-self.r0_ac)))
        return j

    # LEPS (London-Eyring-Polanyi-Sato) potential function - 3 atoms A, B & C confined to motion along a line,
    # Coulomb + exchange interactions of electron clouds
    # Takes 2D list of args (r_ab, r_bc). B is the central atom
    def leps_pot(self, x):
        coulvalue = 0.0
        exchvalue = 0.0
        pairs = ['ab', 'bc', 'ac']
        params = [self.a, self.b, self.c]
        x = np.append(x, x[1]-x[0])
        for i, pair in enumerate(pairs):
            q = self.coul(x[i], pair)
            coulvalue += q
            j1 = np.square(self.exchange(x[i], pair)) / np.square(1.0 + params[i])
            exchvalue += j1
            try:
                j2 = self.exchange(x[i], pair)*self.exchange(x[i+1], pairs[i+1]) / ((1.0 + params[i])*(1.0 + params[i+1]))
            except IndexError:
                j2 = self.exchange(x[i], pair)*self.exchange(x[0], pairs[0]) / ((1.0 + params[i])*(1.0 + params[0]))
            exchvalue -= j2
        potvalue = coulvalue - (exchvalue)**(1.0/2.0)
        return potvalue

    # LEPS potential (see above) with fixed endpoint atoms, with fourth atom coupled in a harmonic way to the central atom B
    # Is used as a simple representation of an activated process coupled to a medium, such as a chemical reaction in a liquid
    # solution or on a solid matrix
    # Takes 2D list of args (r_ab, x). r_ac is a fixed parameter.
    def leps_ho_pot(self, x):
        leps_val = self.leps_pot((x[0], self.r_ac - x[0]))
        ho_val = 2*self.k_c*(x[0] - ((self.r_ac/2.) - (x[1]/self.c))**2)
        return leps_val + ho_val

    # Wolfe-Quapp potental function - expects 2D list of args
    # Minima at (1.124, -1.486) and (-1.174, 1.477)
    # TS at (0.093, 0.174)
    # TP at (1.849, 0.635)
    # VRT at (1.842, 0.768)
    def wq_pot(self, x):
        return x[0]**4 + x[1]**4 - (2.0*np.square(x[0])) - (4.0*np.square(x[1])) + (x[0]*x[1]) + (0.3*x[0]) + (0.1*x[1])

    # Neria-Fischer-Karplus potential function - expects 2D list of args
    # Minima at (+/- 2.71, -/+ 0.15) with energy -5.25.
    # Saddle point at (0,0) with energy -0.002.
    # Valley-ride transition (VRT) point at (0.002, -1.28)
    # Turning point (TP) at (0.871, -1.428)
    def karplus_pot(self, x):
        return 0.06*np.square(np.square(x[0]) + np.square(x[1])) + (x[0]*x[1]) - \
               (9.0*(np.exp(-np.square(x[0]-3.0)-np.square(x[1])) + np.exp(-np.square(x[0]+3.0)-np.square(x[1]))))

    # Three-hole potential - expects 2D list of args
    def three_hole_pot(self, x):
        return 3.*np.exp(-x[0]**2-((x[1]-(1./3.))**2)) - 3.*np.exp(-x[0]**2-((x[1]-(5./3.))**2)) \
               - 5.*np.exp(-(x[0]-1.)**2-x[1]**2) - 5.*np.exp(-(x[0]+1.)**2-x[1]**2) \
               + 0.2*x[0]**4 + 0.2*(x[1]-(1./3.))**4
