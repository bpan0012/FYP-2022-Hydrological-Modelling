import numpy as np

#! HBV


def HBV(Rain, PET, params, n, area):
    # model parameters

    lam = params[0]
    Smax = params[1]
    b = params[2]
    alpha = params[3]
    P = params[4]
    beta = params[5]
    gamma = params[6]
    S2max = params[7]
    k1 = params[8]
    k2 = params[9]

    # model variables
    St = np.zeros([n, 1])
    St[0] = 0
    S1 = np.zeros([n, 1])
    S1[0] = 0
    S2 = np.zeros([n, 1])
    S2[0] = 0
    Qmod = np.zeros([n, 1])
    Rtot = Rain
    time_factor = 24 * 60 * 60
    # Loopping for each day (n)
    for i in range(n - 1):
        ETR = 1 / lam * St[i] / Smax * PET[i]
        Rin = (1 - St[i] / Smax)**b * Rtot[i]
        Reff = Rtot[i] - Rin
        Perc = P * (1 - np.exp(-beta * St[i] / Smax))
        St[i + 1] = np.minimum(np.maximum(St[i] + (Rin - ETR - Perc) * time_factor, 0), Smax)
        R2 = alpha * St[i] / Smax * Reff
        Q2 = k2 * (S2[i] / S2max)**gamma
        S2[i + 1] = np.max(S2[i] + (R2 - Q2) * time_factor, 0)
        R1 = Reff - R2
        Q1 = k1 * S1[i]
        S1[i + 1] = np.maximum(S1[i] + (R1 - Q1 + Perc) * time_factor, 0)
        Qmod[i] = Q1 + Q2
    return Qmod


def HBV_ensemble(Rain, PET, params, ensem_params, n, area):
    # ensem_params = [[mean,sttd,1/0 for ensemble]....]
    n_ensemble = 50
    Qmod_ensemble = np.zeros([n, n_ensemble])
    Rtot = Rain
    time_factor = 24 * 60 * 60
    # i=4
    # print("mean = "+str(ensem_params[i,0])+", std = "+str(ensem_params[i,1]))
    for j in range(n_ensemble):
        final_params = np.zeros(len(params))
        for i in range(len(params)-1):
            if ensem_params[i, 2] == 1:
                final_params[i] = np.maximum(np.random.normal(ensem_params[i, 0], ensem_params[i, 1], 1), 0)
                # print(final_params[i])
            else:
                final_params[i] = params[i]
        # model parameters
        lam = final_params[0]
        Smax = final_params[1]
        b = final_params[2]
        alpha = final_params[3]
        P = final_params[4]
        beta = final_params[5]
        gamma = final_params[6]
        S2max = final_params[7]
        k1 = final_params[8]
        k2 = final_params[9]
        # storage_start = final_params[10]

        # model variables
        St = np.zeros([n, 1])
        St[0] = 0  # storage_start*Smax
        S1 = np.zeros([n, 1])
        S1[0] = 0
        S2 = np.zeros([n, 1])
        S2[0] = 0

        for i in range(n - 1):
            ETR = 1 / lam * St[i] / Smax * PET[i]
            Rin = (1 - St[i] / Smax)**b * Rtot[i]
            Reff = Rtot[i] - Rin
            Perc = P * (1 - np.exp(-beta * St[i] / Smax))
            St[i + 1] = np.minimum(np.maximum(St[i] + (Rin - ETR - Perc) * time_factor, 0), Smax)
            R2 = alpha * St[i] / Smax * Reff
            Q2 = k2 * (S2[i] / S2max)**gamma
            S2[i + 1] = np.minimum(np.max(S2[i] + (R2 - Q2) * time_factor, 0),
                                   Smax)
            R1 = Reff - R2
            Q1 = k1 * S1[i]
            S1[i + 1] = np.minimum(
                np.maximum(S1[i] + (R1 - Q1 + Perc) * time_factor, 0), Smax)
            Qmod_ensemble[i, j] = Q1 + Q2
    return Qmod_ensemble


def HBVRainEnsemble(Rain, PET, params, n, Area, n_ensemble=50):
    """
    params = ["lam", "Smax", "b", "alpha", "Perc", "beta", "gamma", "S2max", "k1", "k2",mu,sigma,alpah_correl ]
    """
    mu =  1
    sigma = params[10]
    alpha_correl = params[11]
    model_params = params[0:10]

    Q_rain_ensemble = np.ones((n, n_ensemble))
    e_pbase = np.zeros((n, 1))
    e_pbase[0] = 0
    for j in range(n-1):
        e_pbase[j+1] = mu+alpha_correl*(e_pbase[j]-mu)+np.random.normal()*sigma*np.sqrt(1-alpha_correl**2)
    e_p1 = mu+alpha_correl*(e_pbase-mu)
    e_p2 = sigma*np.sqrt(1-alpha_correl**2)
    for i in range(n_ensemble):
        # e_p = createNoise(mu, sigma, alpha_correl, n)
        # e_p = np.zeros((n, 1))
        # e_p[0] = 0
        # for j in range(n-1):
        #     e_p[j+1] = mu+alpha_correl*(e_p[j]-mu)+np.random.normal()*sigma*np.sqrt(1-alpha_correl**2)
        # print(e_p)
        # e_p = np.random.lognormal(mu, sigma, [n, 1])  # alternate
        # e_p = np.log(e_p)

        e_p = e_p1+np.random.normal(size=[n, 1])*e_p2  # main

        R_ensemble = np.multiply(e_p[:, 0], Rain)
        R_ensemble *= (R_ensemble > 0)  # in-place zero-ing
        Qcurr = HBV(R_ensemble, PET, model_params, n, Area)
        Q_rain_ensemble[:, i] = np.squeeze(Qcurr, 1)

    return Q_rain_ensemble

#! HYMOD


def HYMODpower(X, Y):
    X = abs(X)  # Needed to capture invalid overflow with netgative values
    return X**Y


def HYMODexcess(x_loss, cmax, bexp, Pval, PETval):
    # this function calculates excess precipitation and evaporation
    xn_prev = x_loss
    ct_prev = cmax * \
        (1 - HYMODpower((1 - ((bexp + 1) * (xn_prev) / cmax)), (1 / (bexp + 1))))
    # Calculate Effective rainfall 1
    ER1 = max((Pval - cmax + ct_prev), 0.0)
    Pval = Pval - ER1
    dummy = min(((ct_prev + Pval) / cmax), 1)
    xn = (cmax / (bexp + 1)) * (1 - HYMODpower((1 - dummy), (bexp + 1)))

    # Calculate Effective rainfall 2
    ER2 = max(Pval - (xn - xn_prev), 0)

    # Alternative approach
    evap = (1 - (((cmax / (bexp + 1)) - xn) / (cmax / (bexp + 1)))) * \
        PETval  # actual ET is linearly related to the soil moisture state
    xn = max(xn - evap, 0)  # update state

    return ER1, ER2, xn


def HYMODlinearReservoir(x_slow, inflow, Rs):
    # Linear reservoir
    x_slow = (1 - Rs) * x_slow + (1 - Rs) * inflow
    outflow = (Rs / (1 - Rs)) * x_slow
    return x_slow, outflow


def HYMOD(Rain, PET, params, n, area):
    cmax = params[0]
    bexp = params[1]
    alpha = params[2]
    ks = params[3]
    kq = params[4]
    # C= np.zeros([n,1])
    # C[0]=0
    lt_to_m = 1
    Pall = Rain
    # HYMOD PROGRAM IS SIMPLE RAINFALL RUNOFF MODEL
    x_loss = 0.0
    # Initialize slow tank state
    # value of 0 init flow works ok if calibration data starts with low discharge
    x_slow = 2.3503 / (ks * 22.5)
    # Initialize state(s) of quick tank(s)
    x_quick = np.zeros(3)
    t = 0
    outflow = np.zeros([n, 1])
    output = np.zeros([n, 1])
    # START PROGRAMMING LOOP WITH DETERMINING RAINFALL - RUNOFF AMOUNTS
    for t in range(n):
        Pval = Pall[t]
        PETval = PET[t]
        # Compute excess precipitation and evaporation
        ER1, ER2, x_loss = HYMODexcess(x_loss, cmax, bexp, Pval, PETval)
        # Calculate total effective rainfall
        ET = ER1 + ER2
        #  Now partition ER between quick and slow flow reservoirs
        UQ = alpha * ET
        US = (1 - alpha) * ET
        # Route slow flow component with single linear reservoir
        x_slow, QS = HYMODlinearReservoir(x_slow, US, ks)
        # Route quick flow component with linear reservoirs
        inflow = UQ

        for i in range(3):
            # Linear reservoir
            x_quick[i], outflow = HYMODlinearReservoir(x_quick[i], inflow, kq)
            inflow = outflow

        # Compute total flow for timestep
        output[t] = ((QS + outflow) / lt_to_m)

    return output


def HYMOD_ensemble(Rain, PET, params, ensem_params, n, area):
    n_ensemble = 250
    Qmod_ensemble = np.zeros([n, n_ensemble])
    Pall = Rain
    # i=4
    # print("mean = "+str(ensem_params[i,0])+", std = "+str(ensem_params[i,1]))
    for j in range(n_ensemble):
        final_params = np.zeros(len(params))
        for i in range(len(params)-1):
            if ensem_params[i, 2] == 1:
                final_params[i] = np.maximum(
                    np.random.normal(ensem_params[i, 0], ensem_params[i, 1], 1), 100)
                # print(final_params[i])
            else:
                final_params[i] = params[i]
        # model parameters
        cmax = final_params[0]
        bexp = final_params[1]
        alpha = final_params[2]
        ks = final_params[3]
        kq = final_params[4]
        C = np.zeros([n, 1])
        C[0] = 0

        # HYMOD PROGRAM IS SIMPLE RAINFALL RUNOFF MODEL
        x_loss = 0.0
        # Initialize slow tank state
        # value of 0 init flow works ok if calibration data starts with low discharge
        x_slow = 2.3503 / (ks * 22.5)
        # Initialize state(s) of quick tank(s)
        x_quick = np.zeros(3)

        outflow = np.zeros([n, 1])

        # START PROGRAMMING LOOP WITH DETERMINING RAINFALL - RUNOFF AMOUNTS
        for i in range(n):
            Pval = Pall[i]
            PETval = PET[i]
            # Compute excess precipitation and evaporation
            ER1, ER2, x_loss = HYMODexcess(x_loss, cmax, bexp, Pval, PETval)
            # Calculate total effective rainfall
            ET = ER1 + ER2
            #  Now partition ER between quick and slow flow reservoirs
            UQ = alpha * ET
            US = (1 - alpha) * ET
            # Route slow flow component with single linear reservoir
            x_slow, QS = HYMODlinearReservoir(x_slow, US, ks)
            # Route quick flow component with linear reservoirs
            inflow = UQ

            for k in range(3):
                # Linear reservoir
                x_quick[k], outflow = HYMODlinearReservoir(
                    x_quick[k], inflow, kq)
                inflow = outflow

            # Compute total flow for timestep
            Qmod_ensemble[i, j] = ((QS + outflow))

    return Qmod_ensemble
