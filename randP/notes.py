## drawing tau
# if true:
#     delta_init = 1^2*tau
#     %P_init needs to be a randomized draw so that norm(LP) is about the right size.
#     P_init = randP(delta_init,tauKtK,tauKtS,LtL,nt);
#     a0 = 1;
#     b0 = (1+nt/2)/delta_init - 0.5*norm(L*P_init)^2;
# else:
    a0 = 0.01
    b0 = 1e-6


# tau = 1/(sigma**2)
# alpha = comes from regularization solution
# delta_init = alpha^2*tau
# P_init = randP(delta_init,tauKTK,tauKtS,LtL,nr)

, init = 'adapt_diag'