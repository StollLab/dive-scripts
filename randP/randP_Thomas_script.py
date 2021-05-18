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

# %%
from pymc3.step_methods.arraystep import BlockedStep

class SampleRandP(BlockedStep):
    def __init__(self, var, delta, sigma, KtK, KtS, LtL, nr):
            self.vars = [var]
            self.var = var
            self.delta = delta
            self.sigma = sigma
            self.KtK = KtK
            self.KtS = KtS
            self.LtL = LtL
            self.nr = nr

    def step(self, point: dict):
        # sigma = np.exp(point[self.sigma.transformed.name])
        sigma = self.sigma
        tau = 1/(sigma**2)
        delta = np.exp(point[self.delta.transformed.name])
       
        new = point.copy()
        new[self.var.name] = dive.randP(delta,tau,self.KtK,self.KtS,self.LtL,self.nr)

        return new

#%%
nr = 100
nt = 100

t = np.linspace(-0.1,2.5,nt)        # time axis, µs
r = np.linspace(2,8,nr)      # distance axis, ns

param2 = [4, 0.3,0.6, 4.8, 0.5, 0.4] # parameters for Gaussian model
bimodal = [4, 0.3,0.6, 6, 0.5, 0.4] # parameters for Gaussian model
Ptrue = dl.dd_gauss2(r,bimodal)  

lam = 0.5
k = 0.1
V0 = 1                      # modulation depth

B = dive.bg_exp(t,k)         # background decay
K = dl.dipolarkernel(t,r,integralop=False)    # kernel matrix
# dr_ = np.zeros(len(r))
# dr_[0:-1] = r[1:] - r[0:1]
# dr_ = dr_/2
# K = K*dr_


S0 = K@Ptrue
sigma = 0.025*max(S0)
S = S0 + dl.whitegaussnoise(t,sigma,seed=0)

fig, ax = plt.subplots(1,2)
line0 = ax[0].plot(t, S0)
line1 = ax[0].plot(t, S)
line2 = ax[1].plot(r, Ptrue)

ax[0].set(xlim = [min(t),max(t)], xlabel = 't (µs)', ylabel = 'S')
ax[1].set(xlim = [min(r),max(r)], xlabel = 'r (nm)', ylabel = 'P')
plt.tight_layout()
plt.show()

#%%
a0 = 0.01
b0 = 1e-6

# tau = 1/(sigma**2)
# alpha = comes from regularization solution
# delta_init = alpha^2*tau
# P_init = randP(delta_init,tauKTK,tauKtS,LtL,nr)

KtK = np.matmul(np.transpose(K),K)
KtS = np.matmul(np.transpose(K),S)

rn = np.linspace(1,nr,nr) 
L = dl.regoperator(rn,1)
LtL = np.matmul(np.transpose(L),L)

# These parameters are only used for the definition of P0 as a pymc3 variable
Pmap = Ptrue

scale = max(S)
tau = 1/sigma**2

#%%
with pm.Model() as model:
    # Noise
    # sigma = pm.Gamma('sigma', alpha=0.7, beta=2)
    # tau = pm.Deterministic('tau',1/(sigma**2))
    

    # Regularization parameter
    # a = a0 + nr/2
    # b = b0 + (1/2)*T.sum(pm.math.dot(L,P0)**2)
    a = a0
    b = b0
    delta = pm.Gamma('delta', alpha=a, beta=b)
    lg_alpha = pm.Deterministic('lg_alpha',np.log10(np.sqrt(delta/tau)))
    
    # Distribution
    invSigma = (tau*KtK+delta*LtL)
    Sigma = tnp.matrix_inverse(invSigma)
    C_L = snp.cholesky(Sigma)
    P = pm.MvNormal("P", mu=Pmap, chol = C_L, shape = nr)    
    # P = pm.Deterministic("P",P0/T.sum(P0)/(r[1]-r[0]))

    # Time domain
    V0 = pm.Normal('V0', mu=1, sigma=0.2)
    Smodel = pm.math.dot(K,P)

    # Likelihood
    pm.Normal('S',mu = V0*Smodel, sigma = sigma, observed = S)
    # pm.Normal('S',mu = Smodel, sigma = sigma, observed = S)

##%
with model:
    step_P = SampleRandP(P, delta, sigma, KtK, KtS, LtL, nr)  
    trace = pm.sample(step = step_P, chains=4, cores=1, draws=7000, tune=4000,return_inferencedata=False)

pm.save_trace(trace = trace, directory = './scripts/randP/edwards_bimodal.trace',overwrite=True)
##%
dive.summary(trace, model, S, t, r, Ptrue = Pmap)

