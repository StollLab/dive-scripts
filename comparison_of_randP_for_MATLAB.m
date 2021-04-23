nr = 50;
nt = 100;

t = linspace(-0.1,2,nt);
r = linspace(2,6,nr);
r0 = 4;
fwhm = 0.2;
P0 = gaussianmodel(r0,fwhm,1,r);
lambda = 0.5;
k = 0.1;
V0 = 1;

B = exp(-abs(t)*k);     
K = dipolarkernel(t,r); 

V = V0*(1-lambda+lambda*K*P0).*B';
sigma = 0.01;
Vexp = addnoise(V,1/sigma,'n');

figure(1)
clf
subplot(1,2,1)
plot(t,V,t,Vexp)
xlabel('t')
ylabel('V')
subplot(1,2,2)
plot(r,P0)
xlabel('r')
ylabel('P')
%%
delta = 0.01;
tau = 1/sigma^2;

tauKtK = tau*(K.'*K);
S = K*P0;

tauKtS = tau*K.'*S;

L = regop(r,1);
LtL = L.'*L;

P = randP(delta,tauKtK,tauKtS,LtL,nr);

figure(2)
clf
plot(r,P)
xlabel('r')
ylabel('P')

%%


function P = randP(delta,tauKtK,tauKtS,LtL,nt)
% based on:
% J.M. Bardsley, C. Fox, An MCMC method for uncertainty quantification in
% nonnegativity constrained inverse problems, Inverse Probl. Sci. Eng. 20 (2012)
invSigma = tauKtK + delta*LtL;
try
  % 'lower' syntax is faster for sparse matrices. Also matches convention in
  % Bardsley paper.
  C_L = chol(inv(invSigma),'lower');
catch
  C_L = sqrtm(inv(invSigma));
end
v = randn(nt,1);
w = C_L.'\v;

P = fnnls(invSigma,tauKtS+w);

end