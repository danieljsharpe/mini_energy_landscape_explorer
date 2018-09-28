'''
Python functions for L-BFGS minimisation algorithm using optional (strong) Wolfe conditions line search
'''

from __future__ import division
from copy import deepcopy
import numpy as np

# A simple test function (2D Rosenbrock) that takes a 2D list of args. Single min. f(x) = 0.0 at x[0] = x[1] = 1.0
def rosenbrock_2d(x):
    return 100.0*np.square(np.square(x[0])-x[1]) + np.square(x[0]-1)

# analytical deriv of 2D Rosenbrock function
def drosenbrock_2d(x, magn=False):
    df1 = 400.0*x[0]*(np.square(x[0])-x[1]) + 2.0*(x[0]-1.0)
    df2 = -200.0*(np.square(x[0])-x[1])
    if not magn:
        return np.array([df1, df2])
    else:
        return np.linalg.norm(np.array([df1, df2]))

# analytical second deriv (Hessian matrix) of 2D Rosenbrock function
def ddrosenbrock_2d(x):
    df11 = 1200.0*np.square(x[0]) - 400.0*x[1] + 2.0
    df22 = 200.0
    df12 = -400.0*x[0]
    return np.array([[df11, df12], [df12, df22]])

# simple numerical deriv of potential energy function by central difference formula
def cd_deriv(func, x, h=1.0E-3, partial=True):
    if not partial:
        x1 = map(lambda x:x+h, x)
        x2 = map(lambda x:x-h, x)
        return ((func(x1)-func(x2))/(2.0*h))
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

def goldsection_ls():
    return

# Function uses interpolation to find a step size value alpha_x such that: alpha_lo < alpha_x < alpha_hi
# and 
# At this point, alpha_lo is, among all step sizes generated so far that satisfy the sufficient decrease
# (Armijo) condition, that which gives the smallest function value
def zoom(func, dfunc, x0, p, alpha_lo, alpha_hi, phi_x_lo, c1, c2, strong):
    phi_x0 = func(x0)
    dphi_x0 = dfunc(x0)
    phi_x_lo = func(x0 + alpha_lo*p)
    i = 0
    while True:
        alpha_x = 0.5*(alpha_lo + alpha_hi)
        xx = x0 + alpha_x*p
        phi_xx = func(xx)
        if (phi_xx > phi_x0 + c1*alpha_x*np.dot(np.transpose(p), dphi_x0)) or \
                phi_xx >= phi_x_lo:
            alpha_hi = alpha_x
        else:
            dphi_xx = dfunc(xx)
            if not strong:
                if -np.dot(np.transpose(p), dphi_xx) <= -c2*np.dot(np.transpose(p), dphi_xp):
                    return alpha_x
            else:
                if abs(np.dot(np.transpose(p), dphi_xx)) <= -c2*np.dot(np.transpose(p), dphi_x0):
                    return alpha_x
            if np.dot(np.transpose(p), dphi_xx)*(alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha_x
            phi_x_lo = phi_xx
        i += 1
    return

# Function to perform Wolfe line search, with strong or weak conditions
# Note: 0 < c1 < c2 < 1. This guarantees that, if the Armijo condition is not satisfied, there
# must be an acceptable point between alpha_p and alpha_x
# given a search direction p, starting point x0 and and max allowed step length alpha_max...
# Ref (pseudocode): Nocedal and Wright: Numerical Optimisation
def wolfe_ls(func, dfunc, x0, p, maxiter=None, alpha_max=100.0, c1=1.0E-03, c2=0.3, \
             strong=True):

    alpha_p = 0.0 # step length for prev iter
    alpha_x = alpha_max*np.random.uniform(0,1) # step length for current iter
    i = 0
    phi_x0 = func(x0) # function value at starting point
    dphi_x0 = dfunc(x0) # grad at starting point
    phi_xp = phi_x0 # function value at prev iter
    while True:
        xx = x0 + alpha_x*p
        phi_xx = func(xx) # function value at current iter
        # check if Armijo (sufficient decrease) rule violation OR function increase
        if (phi_xx > phi_x0 + c1*alpha_x*np.dot(np.transpose(p), dphi_x0)) or \
               (phi_xx >= phi_xp and i > 0):
            return zoom(func, dfunc, x0, p, alpha_p, alpha_x, phi_xp ,c1, c2, strong)
        dphi_xx = dfunc(xx) # grad at current iter
        if not strong:
            # check if weak curvature condition satisfied
            if -np.dot(np.transpose(p), dphi_xx) <= -c2*np.dot(np.transpose(p), dphi_x0):
                return alpha_x
        else:
            # check if strong curvature condition satisfied
            if abs(np.dot(np.transpose(p), dphi_xx)) <= -c2*abs(np.dot(np.transpose(p), dphi_x0)):
                return alpha_x
        if np.dot(np.transpose(p), dphi_xx) >= 0: # check if overstepped minimum
                return zoom(func, dfunc, x0, p, alpha_x, alpha_p, phi_xx, c1, c2, strong)
        phi_xp = phi_xx
        dphi_xp = dphi_xx
        alpha_x = alpha_x + (alpha_max-alpha_x)*np.random.uniform(0,1)
        i += 1
        if maxiter is not None and i >= maxiter:
            print "Wolfe line search exceeded max. no. of iterations"
            break
    return None

# Function to perform L-BFGS minimisation given function f(x) and its gradient g(x), where x is an n-dim vector
def lbfgs(x, f, g, m=5, tol=1.0E-6, maxiter=100, linesearch='Wolfe'):

    q = g
    for i in range(m):
        alpha = 1.0
    k = 0
    delta = 1.0
    while k < maxiter and delta > tol:
        k += 1
    return

#Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimisation algorithm (quasi-Newton method)
def bfgs(x0, func, dfunc, alpha=None, maxiter=1000, threshold=1.0E-03, linesearch='Wolfe'):

    hess = np.identity(np.shape(x0)[0]) # approximate Hessian, initialised as identity matrix
#    hess11 = np.random.uniform(0,1)
#    hess22 = np.random.uniform(0,1)
#    hess12 = np.random.uniform(0,1)
#    hess = np.array([[hess11, hess12], [hess12, hess22]])
#    hess = ddrosenbrock_2d(x0) # initialise Hessian as exact for x0
    xx = x0
    df_xx = dfunc(xx)
    i = 0
    imtx = np.identity(np.shape(x0)[0])
    while np.linalg.norm(df_xx) > threshold and i < maxiter:
        print "\nhess is", hess
        print "analytical hess is", ddrosenbrock_2d(xx)
        print "grad is", df_xx
        xp = xx
        p = -np.dot(np.linalg.inv(hess), df_xx)
        print "p is", p
        if linesearch == 'Wolfe':
            alpha = wolfe_ls(func, dfunc, xp, p)
            print "alpha is", alpha
        s = alpha*p
        xx = xp + s
        print "xx is", xx
        df_xp = dfunc(xp)
        df_xx = dfunc(xx)
        y = df_xx - df_xp
#        M = np.dot(y, np.transpose(y)) / np.dot(np.transpose(y), xx - s)
#        N = np.dot(np.transpose(df_xp), df_xp) / np.dot(df_xp, np.transpose(s))
#        hess = hess + M + N
        temp1 = imtx - (np.dot(s, np.transpose(y)) / np.dot(np.transpose(y), s ))
        temp2 = imtx - (np.dot(y, np.transpose(s)) / np.dot(np.transpose(y), s ))
        temp3 = np.dot(s, np.transpose(s)) / np.dot(np.transpose(y), s)
        temp4 = np.dot(np.linalg.inv(hess), temp2)
        temp5 = np.dot(temp1, temp4)
        hessinv = temp5 + temp3
        hess = np.linalg.inv(hessinv)
        i += 1
    print "finished after %i iterations" % i
    return xx

def conj_grad():
    return

# Gradient (steepest) descent minimisation
def grad_desc(x0, func, dfunc, alpha=None, bbmethod=False, maxiter=1000, threshold=1.0E-03, \
              linesearch='Wolfe'):
    i = 0
    xx = x0
    df_xx = dfunc(xx)
    while np.linalg.norm(df_xx) > threshold and i < maxiter:
        xp = xx
        df_xx = -dfunc(xp)
        if bbmethod: #Barzilai-Borwein step size scaling method
            xx = xp + alpha*df_xx
            s = xx - xp
            y = dfunc(xx) - dfunc(xp)
            alpha = np.dot(np.transpose(s), y) / np.dot(np.transpose(y), y)
        elif linesearch == 'Wolfe':
            alpha = wolfe_ls(func, dfunc, xp, df_xx)
            xx = xp + alpha*df_xx
        i += 1
    print "finished after %i iterations" % i
    return xx

'''
#TEST CASE
func = rosenbrock_2d
dfunc = drosenbrock_2d
#x0 = np.array([0.6, 0.5])
x0 = np.array([0.7, 0.6])
p = np.array([-0.3, -0.4])
print "start coords", x0
print "at start func, dfunc", func(x0), dfunc(x0), "\n"

#grad. descent using BB step size scaling method
#x_final = grad_desc(x0, func, dfunc, alpha=0.01, bbmethod=True, linesearch=None)
#print "final point", x_final
#print "func, dfunc", func(x_final), dfunc(x_final)
#grad. descent using Wolfe line search
#x_final = grad_desc(x0, func, dfunc, alpha=None, bbmethod=False, linesearch='Wolfe')
#print "final point", x_final
#print "func, dfunc", func(x_final), dfunc(x_final)
#BFGS using Wolfe line search
x_final = bfgs(x0, func, dfunc, alpha=None, linesearch='Wolfe')
print "final point", x_final
print "func, dfunc", func(x_final), dfunc(x_final)
quit()

alpha_good = wolfe_ls(func, dfunc, x0, p, usezoom = False)
print "found appropriate alpha", alpha_good
xls = x0 + alpha_good*p
print "new coords", xls
print "at end func, dfunc", func(xls), dfunc(xls)
'''

#VERY SIMPLE TEST CASE
def quadratic_2d(x):
    return x[0]**2 + x[1]**2

def grad_quadratic_2d(x):
    return np.array([x[0]**2 + 2*x[1],2*x[0] + x[1]**2])

x_soln = bfgs(np.array([3.0, 3.0]), quadratic_2d, grad_quadratic_2d, alpha=None, linesearch='Wolfe')
print "solution: x=", x_soln

