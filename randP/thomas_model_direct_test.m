% import traces.csv with matlab importer, use matrix

t = traces(1,:);
Slownoise = traces(2,:);
Shighnoise = traces(3,:);

figure(10)
clf
plot(t,Slownoise,t,Shighnoise)
xlabel('t (µs)')
ylabel('S')
legend({'\sigma = 0.01', '\sigma = 0.05'})

sigma1 = 0.01;
sigma2 = 0.05;

%%

pars.rngseed = 42;
pars.nt = 150;
pars.nr = 150;
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

[t_sim,V_sim,Vm_sim,r,P_sim,K] = deermodel(pars);

hold on
plot(t_sim,K*P_sim)
legend({'\sigma = 0.01', '\sigma = 0.05' 'Strue'})
%%

Settings.Pmodel = P_sim;
Settings.chains = 4;
Settings.maxIterations = 100000;
Settings.showGUI = true;

%%
figure(11)
clf
low_noise_chains = DEER_MCMC(t,Slownoise.',r,pars.sigma,Settings);


figure(12)
idx = int32(linspace(0,-500,50));
clf
hold on
for i = idx
  plot(r,low_noise_chains.P(:,end+i,1))
end
xlabel('r')
title('lower noise trace')
ylabel('P')

%%
figure(13)
clf
noisy_chains = DEER_MCMC(t,Shighnoise.',r,pars.sigma,Settings);


figure(14)
idx = int32(linspace(0,-500,50));
clf
hold on
for i = idx
  plot(r,noisy_chains.P(:,end+i,1))
end
xlabel('r')
title('noisy trace')
ylabel('P')