from traceback import print_tb
from functions import *
from math import isnan
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import datetime as dt
from pyswarm import pso
import models
import os
# importing selfmade functions
from models import HBV, HBV_ensemble
from models import HYMOD, HYMOD_ensemble


import cProfile
import pstats
import io
"""
This file is used to generate the results for the final report.
Refer to main.py for the code used to calibrate the models

"""
"""
The calibrated catchments are:

seven_creeks
cotter_river  
woadyy_river  
bell_river
jordan_river , okay HYMOD
cudgewa_creek   
macintyre_river  
reid_creek okay
npara_river 
collie_river 


["seven_creeks","cotter_river","woadyy_river","bell_river","jordan_river","cudgewa_creek","macintyre_river","reid_creek","npara_river","collie_river"]


"""
outputs_loc = "./Results/Final_Report/"
catchment_name_list = ["seven_creeks","cotter_river","woadyy_river","bell_river","jordan_river","cudgewa_creek","macintyre_river","reid_creek","npara_river","collie_river"]
error_model_idxs=[2,4,1,2,2,2,3,3,3,2]
catchment_summury_df = pd.DataFrame(columns=["Catchment Name","Area (km2)","Calibration Period","Validation Period"])

for i in catchment_name_list:
    dates_formatted, Rain_input, Qknown, PET_daily, catchment_info = getCatchmentInputs(i)
    Area = float(catchment_info[0])
    catchment_name_print = catchment_info[1][0:-1]

    save_vals=[catchment_name_print,Area,int(catchment_info[2]),int(catchment_info[4])]
    
    catchment_summury_df.loc[len(catchment_summury_df.index)] = save_vals
#save the summmury to a csv
# print(catchment_summury_df)
# catchment_summury_df.to_csv(outputs_loc+"catchment_summary.csv", index=False)

#drawing box plots of calibrated parameter for each catchment
catchment_name_list = ["seven_creeks","cotter_river","woadyy_river","bell_river","jordan_river","cudgewa_creek","macintyre_river","reid_creek","npara_river","collie_river"]

catchment_print_list=[]
for i in catchment_name_list:
    dates_formatted, Rain_input, Qknown, PET_daily, catchment_info = getCatchmentInputs(i)
    catchment_name_print = catchment_info[1][0:-1]
    catchment_print_list.append(catchment_name_print)

catchment_name_df=pd.DataFrame(catchment_print_list,columns=["Name"])
catchment_print_list = catchment_name_df.sort_values("Name").to_numpy()
calibrated_params_df= pd.read_csv(outputs_loc+"parameter_results.csv")
params_list=["lam","Smax","b","alpha","Perc","beta","gamma","S2max","k1","k2","sigma","alpah_correl"]
params_print=["$\lambda$","$S_{max}$","b",r"$\alpha$","Percolation",r"$\beta$",r"$\gamma$","$S_{2 max}$","$k_1$","$k_2$","$\sigma$",r"$\alpha_{correl}$"]

i=0
plot_xlbl = np.linspace(1,10,10)
# print(plot_xlbl)
n_rows=4
n_cols=3
# fig, axs = plt.subplots(n_rows, n_cols, sharex=True,figsize=(12, 6))
# plt.rc('font', size=15)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# # plt.tight_layout(pad=0.1,w_pad=0.1,h_pad=0.1)
# for j in range(n_rows):
#     for k in range(n_cols):
#         ax = axs[j,k]
#         ax.scatter(plot_xlbl,calibrated_params_df[params_list[i]])
#         ax.set_ylabel(params_print[i],fontsize = 13)
#         # ax.tick_label('y',labelrotation=90)
        
#         plt.xticks(plot_xlbl,plot_xlbl.astype(int))
#         i+=1
#         if i>11:
#             i=11



#! print the normal calibration results of both HBV and HYMOD.
date_saved="220924"
norm_val_results_df= pd.DataFrame(columns=["catchmentName","RMSE","NSE"])
error_model_results_df= pd.DataFrame(columns=["catchmentName","RMSE","NSE","Pval","Spread"])
corell_check_df=pd.DataFrame(columns=["catchmentName","Durb watson","Rho1","Rho2","AR2 Check","ARMA Check"])
model_name = "HYMOD" 
# catchment_name_list=["bell_river"]
for i in range(len(catchment_name_list)):
    catchment_name = catchment_name_list[i]
    dates_formatted, Rain_input, Qknown, PET_daily, catchment_info = getCatchmentInputs(catchment_name)
    Area = float(catchment_info[0])
    catchment_name_print = catchment_info[1][0:-1]

    curr_model, curr_model_ensemble, best_params, all_params, all_params_df, params_label = getModelData(model_name, catchment_name)
    # run validation dataset of the model
    startday = int(catchment_info[4])
    n = int(catchment_info[5])
    # * Normal model
    Q_mod_val = curr_model(Rain_input[startday: startday+n], PET_daily[startday: startday+n], best_params, n, Area)
    Q_known_val = Qknown[startday: startday+n]
    rmse= RMSE(Q_mod_val,Q_known_val)
    nse = NSE(Q_mod_val,Q_known_val)
    printNormalResults(Q_mod_val,Q_known_val)
    data_tosave=[catchment_name_print,rmse,nse]
    norm_val_results_df.loc[len(norm_val_results_df.index)] = data_tosave
    
    R_mm=Rain_input*1000/Area*(24*3600)
    save_plot_loc=outputs_loc+f"final_plots_resids/{catchment_name}_norm_residual_{model_name}_val.svg"
    plotResiduals(Q_mod_val, Qknown, R_mm, dates_formatted, startday, n, catchment_name_print, model_name,save_fig=save_plot_loc)

    """
    #! check for corellation
    residuals = Q_known_val-Q_mod_val
    ei_ei_1_sum = np.sum(np.square(residuals[1:len(residuals)]-residuals[0:-1]))
    ei2_sum = np.sum(np.square(residuals))
    durbin_wawtson = ei_ei_1_sum/ei2_sum

    dl = 1.748
    du = 1.789
    #! checking for AR or ARMA model.
    # calculation done mostly using the pandas library
    input_data = Q_mod_val
    Q_df = pd.DataFrame(input_data)
    Q_df_shifted = pd.concat([Q_df, Q_df.shift(1), Q_df.shift(2)], axis=1)
    Q_df_shifted.columns = ['t', 't+1', 't+2']
    Q_corr = Q_df_shifted.corr()["t"].to_numpy()
    # print(Q_df_shifted.corr())
    r1 = Q_corr[1]
    r1_2 = r1**2
    r2 = Q_corr[2]

    if (r1_2 < 0.5*(r2+1)):
        ar2_check="Valid"
    else:
        ar2_check="Not Valid"

    # * ARMA model
    con1=False
    con2=False
    if (np.abs(r2) < np.abs(r1)):
        con1=True

    if r1 > 0:
        if r2 > r1*(2*r1-1):
            con2=True
            
    else:
        if r2 > r1*(2*r1+1):
            con2=True
           
    if con1 ==True and con2 ==True:
        arma_check="Valid"
    else:
        arma_check="Not Valid"

    save_vals_corell=[catchment_name_print,durbin_wawtson,r1,r2,ar2_check,arma_check]
    corell_check_df.loc[len(corell_check_df.index)] = save_vals_corell
    
    alt_loc = f"./parameters/{date_saved}/all_params_pVal_HBV_{catchment_name}_{date_saved}.csv"
    best_idx = error_model_idxs[i]
    rain_model_name = "HBV_likelyhood"
    rain_ensem_model, rain_model_ensemble, rain_best_params, rain_all_params, rain_all_params_df, rain_params_label = getModelData(rain_model_name, catchment_name, alt_loc=alt_loc, best_idx=best_idx)
    startday = int(catchment_info[4])
    n = int(catchment_info[5])
    ensemble_n = 250
    Q_known_val = Qknown[startday: startday+n]
    
    Q_rain_ensemble = models.HBVRainEnsemble(Rain_input[startday: startday+n], PET_daily[startday: startday+n], rain_best_params, n, Area, n_ensemble=ensemble_n)
    ensemble_mean = Q_rain_ensemble.mean(axis=1)
    Q_ensemble_mean = np.expand_dims(ensemble_mean, axis=1)
    rmse = RMSE(Q_ensemble_mean, Q_known_val)
    nse = NSE(Q_ensemble_mean, Q_known_val)
    likelyhood_val = pValObj(Q_rain_ensemble, Q_known_val)
    print(f"Rain Ensemble: RMSE = {rmse}, NSE = {nse}, P Val = {likelyhood_val}")
       
    #* measure quality of spread
    Q_ensem = Q_rain_ensemble
    Qknown_val = Qknown[startday: startday+n]
    ensem_mean = np.expand_dims(Q_ensem.mean(axis=1), axis=1)
    xi_xik = Q_ensem-ensem_mean
    # ensemble spread
    ensp_i = 1/ensemble_n*np.expand_dims(np.sum(np.square(xi_xik)+1e-6, axis=1), axis=1)  # +1e-6 to eliminate zeros
    # skew
    skew_i = 1/ensemble_n*np.sum(np.power(xi_xik/np.sqrt(ensp_i), 3), axis=1)
    # kurtosis
    kurt_i = 1/ensemble_n*np.sum(np.power(xi_xik/np.sqrt(ensp_i), 4)-3, axis=1)


    # comparison with observation
    mse_i = 1/ensemble_n*np.expand_dims(np.sum(np.square(Q_ensem-Qknown_val), axis=1), axis=1).astype(float)
    ensk_i = np.square(ensem_mean-Qknown_val).astype(float)
    # In summury the results show that this is a chatoic model since ensk_i < mse

    # Ensemble verification
    avg_spread = np.mean(ensk_i)/np.mean(ensp_i)
    print(f"Average spread over observations:{avg_spread}")

    time_avg_rmse = np.mean(np.sqrt(ensk_i))/np.mean(np.sqrt(mse_i))
    rhs_eqn = np.sqrt((ensemble_n+1)/(2*ensemble_n))
    # print(f"time avd RMSE: {time_avg_rmse} which should equal: {rhs_eqn}")

    data_tosave=[catchment_name_print,rmse,nse,likelyhood_val,avg_spread]
    error_model_results_df.loc[len(error_model_results_df.index)] = data_tosave
    pVal_val = return_pval_obj(Q_rain_ensemble,Q_known_val)
    print(f"Validation Pval = {pVal_val}")


    save_loc=outputs_loc+f"error_model/{catchment_name}_error_model_{model_name}_val.svg"
    plotEnsembleResults(Q_rain_ensemble, Q_mod_val, Qknown, dates_formatted, startday, n, catchment_name_print, model_name, title="Validation Results", ci=95,save_fig=save_loc)
    """
# print(norm_val_results_df)
# norm_val_results_df.to_csv(outputs_loc+f"hymod_resid/norm_{model_name}_results.csv", index=False)

# print(corell_check_df)
# corell_check_df.to_csv(outputs_loc+f"norm_validation/corell_check.csv", index=False)

# print(error_model_results_df)
# error_model_results_df.to_csv(outputs_loc+f"error_model/error_model_{model_name}_results.csv", index=False)

# plt.show()