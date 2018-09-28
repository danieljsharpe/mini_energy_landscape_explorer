'''
Python class for nudged elastic band (NEB) calculation to find the MEP connecting two endpoints on a PES
'''

from __future__ import division
from collections import namedtuple
from copy import deepcopy
from math import cos
from math import pi
import numpy as np
from model_potentials import Model_pes
from eigen_follow import ev_following
import neb_tools
import matplotlib.pyplot as plt

class Neb(object):

    def __init__(self, points, nint, kf, nruns, tstep=0.005, potential='Muller-Brown', optimiser='qm_verlet', deriv='cendiff', \
                 ndim=2, thresh=1.0E-03, climb=500, addperpspring=False, dneb=False, evfollowing=False, randseed=19, \
                 reoptimise=True):

        self.points = points # nxn-dimensional list ((init_x, fin_x), (init_y, fin_y)...) of endpoint minima
        self.nint = nint # no. of intermediate images in NEB calculation
        self.nimages = nint + 2 # tot. no. of images (incl. initial and final images)
        self.kf = kf # spring constant connecting images in NEB calculation. Can provide either an integer (all
                     # images have same spring const) or an (n-1)-dimensional list (thereby tailoring distribn of images)
        self.nruns = nruns # max. no. of iterations
        self.tstep = tstep # delta_t in velocity Verlet algorithm
        self.ndim = ndim # dimensionality of PES
        self.thresh = thresh # convergence threshold based on norm of matrix containing NEB forces on all images
        self.climb = climb # no. of runs after which the highest energy image is selected to be the climbing image to find the TS
                           # if climb > nruns then the climbing image calculation is not done
        self.addperpspring = addperpspring # use smooth switching function to turn on _|_ component of spring force when path is kinked Y/N
        self.dneb = dneb # use doubly nudged EB Y/N - similar to addperpspring
        self.evfollowing = evfollowing # use eigenvector following (as opposed to CI-NEB) to find TS
        self.reoptimise = reoptimise # reoptimise end points Y/N
        pes = Model_pes(potential)
        np.random.seed(randseed)
        if potential == 'Muller-Brown':
            self.pesfunc = pes.mb_pot
            self.domain = ((-2.0, 1.5), (-1.0, 2.0))
            self.energies = (-200.0, 200.0, 5.0) # min / max / step for energy contours in plot
        elif potential == 'Leps':
            self.pesfunc = pes.leps_pot
            self.domain = ((0.0, 5.0), (0.0, 5.0))
            self.energies = (-10.0, 15.0, 0.25)
        elif potential == 'Karplus':
            self.pesfunc = pes.karplus_pot
            self.domain = ((-3.0, 3.0), (-2.0, 2.0))
            self.energies = (-6.0, 2.0, 0.2)
        elif potential == 'Wolfe-Quapp':
            self.pesfunc = pes.wq_pot
            self.domain = ((-2.0, 2.0), (-2.0, 2.0))
            self.energies = (-12.0, 10.0, 0.2)
        if optimiser == 'grad_desc':
            self.optim = neb_tools.grad_desc
        elif optimiser == 'qm_verlet':
            self.optim = neb_tools.vel_verlet
        elif optimiser == 'qm_euler':
            self.optim = neb_tools.qm_euler
        if deriv == 'cendiff':
            self.gradfunc = neb_tools.cd_deriv
        return

    # create a contour plot using first 2 dimensions
    def plot_soln(self, images, climber, r_ts):
        delta = 0.025
        x = np.arange(self.domain[0][0], self.domain[0][1]+delta, delta)
        y = np.arange(self.domain[1][0], self.domain[1][1]+delta, delta)
        X, Y = np.meshgrid(x, y)
        f = np.zeros((np.shape(X)[0], np.shape(X)[1]))
        for i, (r1, r2) in enumerate(zip(X, Y)):
            for j, (r3, r4) in enumerate(zip(r1, r2)):
                f[i,j] = self.pesfunc((r3, r4))
        values = np.arange(self.energies[0], self.energies[1], self.energies[2])
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['ytick.direction'] = 'out'
        plt.figure()
        cp = plt.contourf(x, y, f, values)
#        plt.clabel(cp,inline=1,fontsize=10)
        images_x = []
        images_y = []
        for i in range(np.shape(images)[0]):
            images_x.append(images[i,0])
            images_y.append(images[i,1])
        plt.plot(images_x, images_y, 'ro')
        if climber is not None:
            plt.plot(images_x[climber], images_y[climber], 'ko')
        elif r_ts is not None:
            plt.plot(images_x[-1], images_y[-1], 'ko')
        plt.title("MEP on PES")
        plt.colorbar()
        plt.show()

    # function to find the two highest energy images, given the set of converged images. The TS lies between these two points
    def find_highen_images(self, images):
        image_en = np.zeros(shape=(self.nimages))
        for i in range(self.nimages):
            image_en[i] = self.pesfunc(images[i])
        image_idcs = np.arange(self.nimages)
        highen_images = sorted(zip(image_en, image_idcs), cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)[0:2]
        return highen_images

    # Given three images connected by harmonic springs, find the additional force on the central image
    def perp_springforce(self, perpforce, connected_images):
        delta_1 = connected_images[-1] - connected_images[1]
        delta_2 = connected_images[1] - connected_images[0]
        cos_phi = np.dot(delta_1, delta_2) / (np.linalg.norm(delta_1)*np.linalg.norm(delta_2))
        perpforce = perpforce*self.switch_func(cos_phi)
        return perpforce

    # smooth switching function to gradually turn on the perpendicular component of the spring force where
    # the path becomes kinked
    def switch_func(self, cos_phi):
        return 0.5*(1.0 + cos(pi*cos_phi))

    # given a set of vector forces on each image, separate into parallel and perpendicular components
    def project(self, forces, images, climbingimage=False, addnl_springforce=False):
        components = namedtuple('components', ['parr', 'perp'])
        if not climbingimage:
            parr = np.zeros(shape=(self.nimages, np.shape(self.points)[0]))
            perp = np.zeros(shape=(self.nimages, np.shape(self.points)[0]))
            for i in range(1, self.nimages-1):
                delta = images[i+1] - images[i-1]
                unit_tan = delta / np.linalg.norm(delta)
                # projection of forces // & _|_ to tangent line of image i. This tangent line is the vector that bisects
                # the angle formed between the vectors (images[j+1] - images[j]) & ([images[j] - images[j-1])
                parr[i] = unit_tan*np.dot(forces[i], unit_tan)
                perp[i] = forces[i] - parr[i]
                if addnl_springforce: # calc contribution of _|_ component of spring force when path becomes kinked
                    perp[i] = self.perp_springforce(perp[i], images[i-1:i+2])
            return components(parr, perp)
        else:
            parr = np.zeros(shape=(1, np.shape(self.points)[0]))
            perp = np.zeros(shape=(1, np.shape(self.points)[0]))
            delta = images[-1] - images[0]
            unit_tan = delta / np.linalg.norm(delta)
            parr[0] = unit_tan*np.dot(forces[0], unit_tan)
            perp[0] = forces[0] - parr[0]
            return components(parr, perp)

    # function to get forces on images resulting from mutual springs. No net spring force acts on the endpoint images
    def image_springforce(self, images):
        springforces = np.zeros(shape=(self.nimages, np.shape(self.points)[0]))
        if type(self.kf) is float: # all springs have same force constant
            for j in range(1, self.nimages-1):
                springforces[j] = self.nimages*self.kf*(2.0*images[j] - images[j-1] - images[j+1])
        elif type(self.kf) is list:
            for j in range(1, self.nimages-1):
                springforces[j] = self.kf[j]*(images[j] - images[j+1]) + self.kf[j-1]*(images[j] - images[j-1])
        return springforces

    # function to get forces on images resulting from potential function
    def image_pesforce(self, images):
        grad_images = np.zeros(shape=(self.nimages, np.shape(self.points)[0]))
        for i in range(self.nimages): # get forces on images
            grad_images[i] = self.gradfunc(self.pesfunc, images[i])
        return grad_images

    def init_vels(self):
        vels = np.random.normal(loc=0.0, scale=0.5, size=(self.nimages, np.shape(self.points)[0]))
        return vels        

    # linear interpolation to find the initial positions of the nint images between endpoint minima
    def lin_interp(self):
        delta = points[1] - points[0]
        images = np.ones(shape=(self.nimages, np.shape(delta)[0]))
        lambdas = np.arange(self.nimages) / (self.nimages - 1.0)
        for i in range(self.nimages):
            images[i] = points[0] + lambdas[i]*delta
        return images

    # main function to drive NEB calculation
    def neb_calc(self):
        if self.reoptimise: # reconverge endpoints
            self.points[0] = neb_tools.newton_raphson(self.pesfunc, self.points[0])
            self.points[1] = neb_tools.newton_raphson(self.pesfunc, self.points[1])
        print "endpoints are:\n", self.points
        images = self.lin_interp()
        climber = None
        if self.optim is neb_tools.vel_verlet or self.optim is neb_tools.qm_euler:
            vels = self.init_vels()
        neb_deriv = np.ones(shape=(self.nimages, np.shape(vels[0])[0]))
        if self.dneb: dneb_force = np.zeros(shape=(self.nimages, np.shape(vels[0])[0]))
        i = 0
        while i < self.nruns and np.linalg.norm(neb_deriv) > self.thresh:
            grad_images = -1.0*self.image_pesforce(images)
            springforces = -1.0*self.image_springforce(images)
            if not self.addperpspring and not self.dneb:
                # Keep only _|_ component of force due to potential function and only // component of force due to springs
                neb_deriv = self.project(grad_images, images).perp + self.project(springforces, images).parr
            elif self.addperpspring: # use a fraction of the perpendicular component of the spring force
                springforce_component = self.project(springforces, images, addnl_springforce=True)
                neb_deriv = self.project(grad_images, images).perp + springforce_component.parr + springforce_component.perp
            elif self.dneb: # uses an additional force that is the component of the _|_ component of the spring force that is
                            # orthogonal to the _|_ component of the potential force
                springforce_component = self.project(springforces, images)
                potforce_component = self.project(grad_images, images)
                for k in range(1, self.nimages-1):
                    dneb_force[k] = springforce_component.perp[k] - np.dot(springforce_component.perp[k], \
                        (potforce_component.perp[k] / np.linalg.norm(potforce_component.perp[k])))
                neb_deriv = potforce_component.perp + springforce_component.parr + dneb_force
            if climber is not None:
                neb_deriv[climber] = self.project(grad_images[climber], images[climber-1:climber+2], \
                        climbingimage=True).perp - \
                        2.0*self.project(grad_images[climber], images[climber-1:climber+2], climbingimage=True).parr
            images, vels = self.optim(images, vels, neb_deriv, self.nimages, self.tstep)
            i += 1
            if i == self.climb:
                climber = self.find_highen_images(images)[0][1]
                print "After %i steps, selected image %i as climber" % (i, climber)
        print "Calculation finished after %i steps" % i
        print "Final images are\n", images
        if climber is None and self.evfollowing:
            r_ts_idx = self.find_highen_images(images)[0][1]
            r_ts = np.array(deepcopy(images[r_ts_idx]))
            print "beginning eigenvector following starting from image at:", r_ts
            r_ts = ev_following(self.pesfunc, r_ts) # eigenvector-following starting from highest energy image
            print "transition state is at:", r_ts
            images = np.append(images, r_ts).reshape((self.nimages+1,self.ndim))
        else: r_ts = None
        self.plot_soln(images, climber, r_ts)
        return


#Driver code

#TEST CASE 1: MULLER-BROWN POT.
#set params
points = np.array([[-0.563, 1.417], [0.618, 0.01]]) # endpoint minima
points = np.array([[-0.55822365, 1.4417258], [0.62349939, 0.02803773]])
nint = 24
kf = 1.0
nruns = 15000
neb1 = Neb(points, nint, kf, nruns, potential = 'Muller-Brown', optimiser = 'qm_euler', climb=5000, dneb=True, \
           addperpspring=False, evfollowing=False)

'''
#TEST CASE 2: WOLFE-QUAPP POT.
points = np.array([[-1.174, 1.477], [1.124, -1.486]])
nint = 14
kf = 1.0
nruns = 15000
neb1 = Neb(points, nint, kf, nruns, potential = 'Wolfe-Quapp', optimiser = 'qm_euler', climb=500, dneb=False, addperpspring=True)
'''
'''
#TEST CASE 3: NERIA-FISCHER-KARPLUS POT.
points = np.array([[-2.71, 0.15], [2.71, -0.15]])
nint = 30
kf = 1.0
nruns = 3000
neb1 = Neb(points, nint, kf, nruns, potential = 'Karplus', optimiser = 'qm_euler', climb=nruns+1, addperpspring=True, \
           evfollowing=True)
'''
'''
#TEST CASE 4: LEPS POT.
points = np.array([[0.7, 4.0], [3.35, 4.09]])
nint = 30
kf = [1.0]*(nint+1)
nruns = 3000
neb1 = Neb(points, nint, kf, nruns, potential='Leps', optimiser='qm_euler', climb=nruns+1, dneb=False, addperpspring=True,
           evfollowing=True, reoptimise=False)
'''
neb1.neb_calc()
