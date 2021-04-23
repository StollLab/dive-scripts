# %%
import numpy as np
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

traces = []

nr = 50
nt = 100

t = np.linspace(-0.1,2,nt)        # time axis, µs
r = np.linspace(2,6,nr)      # distance axis, ns

# param = [4, 0.2] # parameters for three-Gaussian model
# P0 = dl.dd_gauss(r,param)          # model distance distribution

# param3 = [4, 0.6,0.3, 3, 0.6, 0.3, 5.5, 0.6, 0.3] # parameters for three-Gaussian model
# P0 = dl.dd_gauss3(r,param3)          # model distance distribution

param3 = [4, 0.3,0.6, 4.8, 0.5, 0.4] # parameters for three-Gaussian model
Pin = dl.dd_gauss2(r,param3)          # model distance distribution

lam = 0.5
k = 0.1
V0 = 1                      # modulation depth

B = dive.bg_exp(t,k)         # background decay
K = dive.dipolarkernel(t,r)    # kernel matrix

Vexp = dive.deerTrace(K@Pin,B,V0,lam) + dl.whitegaussnoise(t,0.01,seed=0)

# fig, ax = plt.subplots(1,2)
# line1 = ax[0].plot(t, Vexp)
# line2 = ax[1].plot(r, P0)

# ax[0].set(xlim = [min(t),max(t)], xlabel = 't (µs)', ylabel = 'V')
# ax[1].set(xlim = [2,6], xlabel = 'r (nm)', ylabel = 'P')
# plt.tight_layout()
# plt.show()

# %%

L = dl.regoperator(r,d=2)

LtL = np.dot(L.transpose(),L)

KtK = np.dot(K.transpose(),K)

param = [5, 0.8] # parameters for three-Gaussian model
PsingleGaussOff = dl.dd_gauss(r,param)          # model distance distribution

param = [4, 0.8] # parameters for three-Gaussian model
PbroadGauss = dl.dd_gauss(r,param)          # model distance distribution

param3 = [4, 0.6,0.3, 3, 0.6, 0.3, 5.5, 0.6, 0.3] # parameters for three-Gaussian model
PmultiGauss = dl.dd_gauss3(r,param3)          # model distance distribution

Puniform = np.ones(r.shape)
Puniform = Puniform/sum(Puniform)


# %%
# Pmap = Pin
# Pmap = PsingleGaussOff
# Pmap = PmultiGauss
# Pmap = PbroadGauss
Pmap = Puniform


with pm.Model() as model:
    # Noise
    sigma = pm.Gamma('sigma', alpha=0.7, beta=2)

    #  Distribution model
    delta = pm.Gamma('delta', alpha=0.7, beta=2)

    tau = pm.Deterministic('tau',1/(sigma*sigma))
    invSigma = (tau*KtK+delta*LtL)
    # invSigma = (tau*KtK)
    Sigma = tnp.matrix_inverse(invSigma)
    C_L = snp.cholesky(Sigma)

    P0 = pm.MvNormal("P0", mu=Pmap, chol = C_L, shape = nr)        
    # P0 = pm.Bound(pm.MvNormal, lower=0.0)("P0", mu=Pmap, chol = C_L, shape = nr)  
    P = pm.Deterministic("P",P0/T.sum(P0)/(r[1]-r[0]))
    
    # Background
    k = pm.Gamma('k', alpha=0.5, beta=2)
    B = dive.bg_exp(t,k)

    # DEER Signal
    lamb = pm.Beta('lamb', alpha=1.3, beta=2.0)
    V0 = pm.Bound(pm.Normal,lower=0.0)('V0', mu=1, sigma=0.2)

    Vmodel = dive.deerTrace(pm.math.dot(K,P),B,V0,lamb)
    pm.Normal('V',mu = Vmodel, sigma = sigma, observed = Vexp)

trace = pm.sample(model = model,chains=8, cores=1, draws=2000, tune=1000,return_inferencedata=False)
traces.append(trace)
# %%
Ps, Vs, _, _ = dive.drawPosteriorSamples(traces[-1],r,t,100)
dive.plotMCMC(Ps,Vs,Vexp,t,r)
# %%
