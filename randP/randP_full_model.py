import numpy as np
import scipy as sp
import pandas as pd
from os import listdir
from os.path import isfile, join
import pymc3 as pm
import math as m
import arviz as az

import dive
import matplotlib.pyplot as plt
import deerlab as dl

from theano import tensor as T
from theano.tensor import nlinalg as tnp
from theano.tensor import slinalg as snp

nr = 100
nt = 150

t = np.linspace(-0.1,2.5,nt)        # time axis, µs
r = np.linspace(2,8,nr)      # distance axis, ns

r0 = 4 
fwhm = 0.4 # parameters for three-Gaussian model
Ptrue = dive.dd_gauss(r,r0,fwhm)          # model distance distribution

# param2 = [4, 0.3,0.6, 4.8, 0.5, 0.4] # parameters for three-Gaussian model
# Ptrue = dl.dd_gauss2(r,param2)  

lam = 0.5
k = 0.1
V0 = 1                      # modulation depth

B = dl.bg_exp(t,k)         # background decay
K = dl.dipolarkernel(t,r,mod = lam, bg = B)    # kernel matrix

sigma = 0.01

# Vtrue = dive.deerTrace(K@P0,B,V0,lam)
Vtrue = K@Ptrue
Vexp = Vtrue + dl.whitegaussnoise(t,sigma,seed=0)

# Vexp = dive.deerTrace(K@P0,B,V0,lam) + dl.whitegaussnoise(t,sigma,seed=0)

import warnings
import scipy.integrate
from scipy.special import fresnel

def kernelmatrix_fresnel(t,r):

    ge = 2.00231930436256 # free-electron g factor
    muB = 9.2740100783e-24 # Bohr magneton, J/T 
    mu0 = 1.25663706212e-6 # magnetic constant, N A^-2 = T^2 m^3 J^-1 
    h = 6.62607015e-34 # Planck constant, J/Hz
    w0 = (mu0/2)*muB**2*ge*ge/h*1e21 # Hz m^3 -> MHz nm^3 -> Mrad s^-1 nm^3
    
    #==========================================================================
    """Calculate kernel using Fresnel integrals (fast and accurate)"""
    K = np.zeros((nt,nr))
    wr = w0/(r**3)  # rad s^-1

    # Calculation using Fresnel integrals
    ph = np.outer(np.abs(t), wr)
    kappa = np.sqrt(6*ph/np.pi)
    
    # Supress divide by 0 warning        
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        S, C = fresnel(kappa)/kappa
    
    K = C*np.cos(ph) + S*np.sin(ph)
    K[t==0] = 1 

    return K

K = kernelmatrix_fresnel(t,r)    # unnormalized kernel matrix
dr = r[1]-r[0]

rn = np.linspace(1,nr,nr) 
L = dl.regoperator(rn,1)
LtL = np.matmul(np.transpose(L),L)

a0 = 0.01
b0 = 1e-6

Pmap = Ptrue

with pm.Model() as model:
    # Noise
    sigma = pm.Gamma('sigma', alpha=0.7, beta=2)
    tau = pm.Deterministic('tau',1/(sigma**2))

    # time domainnal
    lamb = pm.Beta('lamb', alpha=1.3, beta=2.0)
    V0 = pm.Bound(pm.Normal,lower=0.0)('V0', mu=1, sigma=0.2)
    
    # Background
    k = pm.Gamma('k', alpha=0.5, beta=2)
    B = dl.bg_exp(t,k)         # background decay

    # Calculate matrices and operators
    Kl = (1-lamb)+lamb*K
    Bm = T.transpose(T.tile(B,(nr,1)))
    KB = Kl*Bm*dr
    KtK = T.dot(T.transpose(KB),KB)

    # Regularization parameter
    # a = a0 + nr/2
    # b = b0 + (1/2)*T.sum(pm.math.dot(L,P0)**2)
    a = a0
    b = b0
    delta = pm.Gamma('delta', alpha=a, beta=b)
    alpha = pm.Deterministic('alpha',np.sqrt(delta/tau))
    
    # Distribution
    invSigma = (tau*KtK+delta*LtL)
    Sigma = tnp.matrix_inverse(invSigma)
    C_L = snp.cholesky(Sigma)
    P0 = pm.MvNormal("P0", mu=Pmap, chol = C_L, shape = nr)    
    P = pm.Deterministic("P",P0/T.sum(P0)/(r[1]-r[0]))

    # Likelihood
    pm.Normal('V',mu = V0*pm.math.dot(KB,P), sigma = sigma, observed = Vexp)

from pymc3.step_methods.arraystep import BlockedStep

class SampleRandP(BlockedStep):
    def __init__(self, var, delta, sigma, k, lamb, LtL, t, V, r):
            self.vars = [var]
            self.var = var
            self.delta = delta
            self.sigma = sigma
            self.k = k
            self.lamb = lamb
            self.LtL = LtL
            self.nr = nr
            self.V = V
            self.r = r
            self.t = t

    def step(self, point: dict):
        # transform parameters
        sigma = np.exp(point[self.sigma.transformed.name])
        tau = 1/(sigma**2)
        delta = np.exp(point[self.delta.transformed.name])
        k = np.exp(point[self.k.transformed.name])
        lamb = np.exp(point[self.lamb.transformed.name])

        nr = len(self.r)

        # Background
        B = dl.bg_exp(self.t,k) 
        # Kernel
        K = dl.dipolarkernel(self.t, self.r, mod = lamb, bg = B)

        KtK = np.matmul(np.transpose(K),K)
        KtV = np.matmul(np.transpose(K),self.V)

        new = point.copy()
        new[self.var.name] = dive.randP(delta,tau,KtK,KtV,self.LtL,nr)

        return new

with model:
    step_P0 = SampleRandP(P0, delta, sigma, k, lamb, LtL, t, Vexp, r)  
    trace = pm.sample(step = step_P0, chains=4, cores=1, draws=5000, tune=5000,return_inferencedata=False, start = {'sigma': 0.01, 'lamb': 0.5})

pm.save_trace(trace = trace, directory = './scripts/randP/fullmodel.trace',overwrite=True)

dive.summary(trace, model, Vexp, t, r, Ptrue = Pmap)