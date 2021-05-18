% set some parameters

% unimodal
pars.a = [0.6 0.4]; % relative amplitudes
pars.r0 = [4 4.8]; % nm
pars.w = [0.3 0.5]; % nm

pars.rngseed = 42;
pars.nt = 100;
pars.nr = 100;
pars.tmin = -0.1; % us
pars.tmax = 2.5; % us
pars.rmin = 2; % nm
pars.rmax = 8; % nm
pars.lambda = 0.5;
k = 0.1;
pars.tB = 1/k; % us
pars.sigma = 0.01;
% pars.sigma = 0.025;
pars.sigma = 0.05;
pars.V0 = 1;

[t,V,Vm,r,P,K] = deermodel(pars);

%%

S0 = K*P;

S = S0 + randn(pars.nt,1)*pars.sigma;

figure
subplot(2,1,1)
plot(r,P)
ylabel('P(r)')
xlabel('r (nm)')
axis tight
subplot(2,1,2)
plot(t,S0,t,S)
axis tight
ylabel('V(t)')
xlabel('t (\mus)')
%%
Settings.Pmodel = P;
Settings.chains = 4;
Settings.maxIterations = 100000;
Settings.showGUI = true;
chains = DEER_MCMC(t,S,r,pars.sigma,Settings);

%%
figure(1)
clf
hold on

for i = 1: 20
  x = randi([1 size(chains.P,2)]);
  c = randi([1 4]);
  plot(r,squeeze(chains.P(:,x,c)))
  disp(chains.alpha(x,c))
end

%%


%%
sigma = 0.05
alpha = 1
tau = 1/sigma^2
delta = (alpha^2)*tau

dr = r(2)-r(1);
L = regop(r,2);
LtL = L.'*L;
tauKtK = tau*(K.'*K);
tauKtS = tau*K.'*S;

Pdraw = randP(delta,tauKtK,tauKtS,LtL,pars.nr);

figure(2)
clf
plot(r,Pdraw)
xlabel('r')
ylabel('P')

%%
nsamples = 50000
delta = zeros(1,nsamples);

a0 = 0.01;
b0 = 1e-6;

figure(3)
clf
plot(r,P)

for i = 1 : nsamples
  delta(i) = randdelta(P,a0,b0,pars.nr,L);
end


%%
figure(4)
clf
histogram(delta,'NumBins',100)

disp('max appears to be around 16500, with halfwidth 14000 - 19200')
 

%%
figure(5)
clf
plot(L*P)

norm(L*P)^2

%%

figure(6)
idx = int32(linspace(0,-200,50))


clf
hold on
for i = idx
  plot(r,chains.P(:,end+i,1))
end
xlabel('r')
title('lower noise trace')
ylabel('P')

%%
figure(7)
idx = int32(linspace(0,-200,50))


clf
hold on
for i = idx
  plot(r,noisy_chains.P(:,end+i,1))
end
xlabel('r')
title('noisy trace')
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

function delta = randdelta(P,a0,b0,nt,L)
a_ = a0 + nt/2;
b_ = b0 + (1/2)*norm(L*P)^2;
% precision in distance domain %randraw uses the shape/scale paramaterization, but we use shape and rate.
delta = randraw('gamma',[0,1/b_,a_],1);
end
