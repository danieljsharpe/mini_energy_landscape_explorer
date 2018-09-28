#Functions to perform eigenvector-following; use of a Lagrange approach for finding TSs

import numpy as np
import neb_tools

# Cerjan-Miller function, for testing. If a=1.0, b= 0.5 and c=1.0; then min. is at (0,0) and TSs are at (-1, 0) and (+1, 0)
def cerjan_miller(x):
    a = 1.0
    b = 0.5
    c = 1.0
    return (a - b*np.square(x[1]))*np.square(x[0])*np.exp(-np.square(x[0])) + (0.5*c*np.square(x[1]))

# given a Hessian matrix, return the eigenvalue of smallest magnitude and the corresponding eigenvector
def get_tsvec(hess):
    evals, evecs = np.linalg.eig(hess)
    ts_eval = sorted([abs(i) for i in evals])[0]
    j = np.where((evals==ts_eval) | (evals==-ts_eval))
    return evals[j], evecs[j]

# Main function to drive eigenvector following
def ev_following(pesfunc, image, c=0.1, maxiter=10000, thresh=1.0E-04, min_start=False, opt_lagrange=False):
    r = image
    i = 0
    h = np.zeros(np.shape(image)[0])
    F = np.ones(np.shape(image)[0])
    if min_start: # starting at min - check hessian is positive definite (all pos eigenvalues), then take step away from min
        hess = neb_tools.cd_2deriv(pesfunc, r)
        ts_eval, tsvec = get_tsvec(hess)[0][0], get_tsvec(hess)[1][0]
        if ts_eval < 0: return None # the starting point is not a minimum
        r += tsvec*c
    while i < maxiter and np.linalg.norm(F) > thresh:
        h = np.zeros(np.shape(image)[0])
        lagrange_m = np.zeros(np.shape(image)[0])
        grad = neb_tools.cd_deriv(pesfunc, r)
        hess = neb_tools.cd_2deriv(pesfunc, r)
        evals, evecs = np.linalg.eigh(hess)
        ts_eval = sorted(evals)[0]
        j = np.where(evals==ts_eval)
        F = np.dot(evecs, grad)
        if ts_eval >= 0:
            lagrange_m[j] = ts_eval + abs(F[j] / c)
        elif ts_eval < 0 and not opt_lagrange:
            # not optimal choice of Lagrange multiplier for b_i < 0 regime, but helps retain control of step size
            lagrange_m[j] = (ts_eval / 2.0) + abs(F[j] / c)
        elif ts_eval < 0 and opt_lagrange: # optimal (most efficient) choice of Lagrange multiplier is zero
            lagrange_m[j] = 0.0
        for k in range(np.shape(h)[0]):
            h += (F[k] / (lagrange_m[k] - evals[k]))*evecs[k]
        r += h
        i += 1
    print "finished eigenvector following in %i steps" % i
    return r

'''
#TEST CASE
pesfunc = cerjan_miller
start = [0.2, -0.6]
f0 = pesfunc(start)
print f0
r_ts = ev_following(pesfunc, start)
print "r_ts is\n", r_ts
'''
