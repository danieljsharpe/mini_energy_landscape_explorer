from __future__ import division
from copy import deepcopy
import numpy as np

#Function to take a single step for Quick-min algorithm coupled with an Euler integrator
def qm_euler(images, vels, forces, nimages, tstep):
    for i in range(1, nimages-1):
        proj = np.dot(vels[i], forces[i])
        if proj < 0.0: # zero the velocity if it is anti-// to the force
            vels[i] = np.zeros(shape=(np.shape(images[0])[0]))
        # take an Euler step
        images[i] = images[i] + tstep*vels[i]
        vels[i] = vels[i] + tstep*forces[i]
    return images, vels

#Function to perform Newton-Raphson minimisation, used for reconverging provided endpoints
def newton_raphson(func, x, gamma=0.01, tol=1.0E-04, maxiter=10000):
    i = 0
    print "starting at", x
    g = np.ones(np.shape(x)[0])
    while i < maxiter and abs(np.linalg.norm(g)) > tol:
        g = cd_deriv(func, x)
        hess = cd_2deriv(func, x)
        x = x - (gamma*np.dot(np.linalg.inv(hess), g))
        i += 1
    print "reconverged endpoint in %i steps" % i
    return x

#Function to take a single velocity Quick-min Verlet timestep according to forces acting on the images
def vel_verlet(images, vels, forces):
    
    return images, vels

# numerical deriv of potential energy function by central difference formula
def cd_deriv(func, x, h=1.0E-4, partial=True):
    if not partial:
        x1 = map(lambda x:x+h, x)
        x2 = map(lambda x:x-h, x)
        return (func(x1)-func(x2))/(2.0*h)
    else:
        n = np.shape(x)[0]
        par_derivs = np.zeros(n)
        for i in range(n):
            x1 = deepcopy(x)
            x1[i] = x[i] + h
            x2 = deepcopy(x)
            x2[i] = x[i] - h
            par_derivs[i] = (func(x1) - func(x2)) / (2.0*h)
        return par_derivs

# numerical second deriv of potential energy function by central difference formula
def cd_2deriv(func, x, h=1.0E-4, partial=True):
    if not partial:
        x1 = map(lambda x:x+h, x)
        x2 = map(lambda x:x-h, x)
        return (func(x1) - (2.0*func(x)) + func(x2)) / np.square(h)
    elif partial and np.shape(x)[0] == 2:
        par_2derivs = np.zeros((2,2))
        x1 = deepcopy(x)
        x1[0] = x1[0] + h
        x2 = deepcopy(x)
        x2[0] = x2[0] - h
        dfdx1 = cd_deriv(func, x1)[1]
        dfdx2 = cd_deriv(func, x2)[1]
        par_2derivs[0,1] = (dfdx1 - dfdx2) / (2.0*h)
        par_2derivs[1,0] = par_2derivs[0,1]
        par_2derivs[0,0] = (func(x1) - (2.0*func(x)) + func(x2)) / np.square(h)
        x3 = deepcopy(x)
        x3[1] = x3[1] + h
        x4 = deepcopy(x)
        x4[1] = x4[1] - h
        par_2derivs[1,1] = (func(x3) - (2.0*func(x)) + func(x4)) / np.square(h)
        return par_2derivs
