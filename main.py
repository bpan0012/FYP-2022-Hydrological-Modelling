# importing the model from models.py
from turtle import up
from functions import *
import models

from math import isnan
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import datetime as dt
import cProfile

"""
# !Catchments
seven_creeks - good both  ----pVal----
?cotter_river - good both  ----pVal----
woadyy_river - okay both - rerun  ----pVal----
bell_river - bad both
jordan_river - okay HBV, okay HYMOD
cudgewa_creek - okay both  ----pVal----
macintyre_river - bad RMSE but okay NSE - needs another run with better bounds
reid_creek - bad/okay
npara_river - good  ----pVal----
collie_river - good  ----pVal----

butmaroo_creek

["seven_creeks","cotter_river","woadyy_river","bell_river","jordan_river","cudgewa_creek","macintyre_river","reid_creek","npara_river","collie_river"]
# !Models

HBV
HYMOD

"""


model_cal_run = 0  # to decide if model calibration will occur
folder_name = "./parameters/"
date_saved = "220924"
folder_loc = folder_name+date_saved+"/"

# get the required model inputs
# catchment_list = ["npara_river"]
model_tocal_list = ["HBV"]
catchment_name = "cudgewa_creek"
if model_cal_run == 1:
    model_name = "HBV"
    calibrate_likelyhood(folder_loc, date_saved, model_name, catchment_name)
    # prepCalModel(catchment_list, model_tocal_list, folder_loc, date_saved)
    """------------checking runtime-------------"""
    # cProfile.run('calibrate_likelyhood(folder_loc, date_saved, model_name, catchment_name)', 'likely_cal')
    """------------checking runtime-------------"""

dates_formatted, Rain_input, Qknown, PET_daily, catchment_info = getCatchmentInputs(catchment_name)
Area = float(catchment_info[0])
catchment_name_print = catchment_info[1][0:-1]


model_name = "HBV"  # model_tocal_list[0]
alt_params = f"./parameters/Calibrated_choosen/{model_name}_params_{catchment_name}_all.csv"
bst_idx = 2
curr_model, curr_model_ensemble, best_params, all_params, all_params_df, params_label = getModelData(model_name, catchment_name)

# check calibration results
startday = int(catchment_info[2])
n = int(catchment_info[3])
Q_mod_cal = curr_model(Rain_input[startday: startday+n], PET_daily[startday: startday+n], best_params, n, Area)
Q_known_cal = Qknown[startday: startday+n]
"""------------checking runtime-------------"""
# cProfile.run('curr_model(Rain_input[startday: startday+n], PET_daily[startday: startday+n], best_params, n, Area)')
"""------------checking runtime-------------"""
# printNormalResults(Q_mod_cal,Q_known_cal)
# plotResultsPercipBar(Q_mod_cal, Qknown, Rain_input, dates_formatted, startday, n, catchment_name_print, model_name, title=", Calibration")

# run a validation run of the model
startday = int(catchment_info[4])
n = int(catchment_info[5])

# * Normal model
Q_mod_val = curr_model(Rain_input[startday: startday+n], PET_daily[startday: startday+n], best_params, n, Area)
Q_known_val = Qknown[startday: startday+n]
printNormalResults(Q_mod_val, Q_known_val, optional_starter="Validation")
# plotResultsPercipBar(Q_mod_val, Qknown, Rain_input, dates_formatted, startday, n, catchment_name_print, model_name, title=", Validation")
residuals = Q_known_val-Q_mod_val

# * model ensemble
# * manual mu/sigma
# params_ensemble_rain = np.append(best_params,[1,0.5,0.5])
# Q_rain_ensemble = models.HBVRainEnsemble(Rain_input[startday: startday+n], PET_daily[startday: startday+n], params_ensemble_rain, n, Area, n_ensemble=250)
# ensemble_mean = Q_rain_ensemble.mean(axis=1)
# Q_ensemble_mean = np.expand_dims(ensemble_mean, axis=1)
# rmse = RMSE(Q_ensemble_mean, Q_known_val)
# nse = NSE(Q_ensemble_mean, Q_known_val)

# likelyhood_val = pValObj(Q_rain_ensemble, Q_known_val)
# print(f"Rain Ensemble: RMSE = {rmse}, NSE = {nse}, P Val = {likelyhood_val}")
# plotEnsembleResults(Q_rain_ensemble, Q_mod_val, Qknown, dates_formatted, startday, n, catchment_name_print, model_name, title="Percipitation-Manual", ci=95)
# * ensemble of percipitation
print("----Input Ensemble-----")
alt_loc = f"./parameters/{date_saved}/all_params_pVal_HBV_{catchment_name}_{date_saved}.csv"
best_idx = 2
rain_model_name = "HBV_likelyhood"
rain_ensem_model, rain_model_ensemble, rain_best_params, rain_all_params, rain_all_params_df, rain_params_label = getModelData(
    rain_model_name, catchment_name, alt_loc=alt_loc, best_idx=best_idx, print_vals=True)

# check what the values are in the obj
startday = int(catchment_info[2])
n = int(catchment_info[3])
idx_List = [best_idx]
for i in idx_List:
    rain_ensem_model, rain_model_ensemble, rain_best_params, rain_all_params, rain_all_params_df, rain_params_label = getModelData(
    rain_model_name, catchment_name, alt_loc=alt_loc, best_idx=i, print_vals=False)
    # !print(rain_best_params)
    rain_best_params[0]=1e-10
    Q_rain_ensemble = models.HBVRainEnsemble(Rain_input[startday: startday+n], PET_daily[startday: startday+n], rain_best_params, n, Area, n_ensemble=50)
    """------------checking runtime-------------"""
    # cProfile.run('models.HBVRainEnsemble(Rain_input[startday: startday+n], PET_daily[startday: startday+n], rain_best_params, n, Area, n_ensemble=100)')
    """------------checking runtime-------------"""
    Q_known_check = Qknown[startday: startday+n]
    ensemble_mean = Q_rain_ensemble.mean(axis=1)
    Q_ensemble_mean = np.expand_dims(ensemble_mean, axis=1)
    p_val_check, nse_check = check_pval(Q_rain_ensemble, Q_known_check)
    rmse = RMSE(Q_ensemble_mean, Q_known_check)
    nse = NSE(Q_ensemble_mean, Q_known_check)
    likelyhood_val = pValObj(Q_rain_ensemble, Q_known_check)
    print(f"Calibration Ensemble: RMSE = {rmse}, NSE = {nse}, P Val = {likelyhood_val}")
    print(f"During calibration the obj values at the end are: percent in CL={p_val_check}, RMSE = {nse_check}")
    pVal_cal = return_pval_obj(Q_rain_ensemble,Q_known_check)
    print(f"Calibration Pval = {pVal_cal}")
    print("---------")
    # plotEnsembleResults(Q_rain_ensemble, Q_mod_cal, Qknown, dates_formatted, startday, n, catchment_name_print, model_name, title="Cal Percip", ci=95)

# * Validation
#! print(rain_best_params)
startday = int(catchment_info[4])
n = int(catchment_info[5])
ensemble_n = 250

Q_rain_ensemble = models.HBVRainEnsemble(Rain_input[startday: startday+n], PET_daily[startday: startday+n], rain_best_params, n, Area, n_ensemble=ensemble_n)
ensemble_mean = Q_rain_ensemble.mean(axis=1)
Q_ensemble_mean = np.expand_dims(ensemble_mean, axis=1)
rmse = RMSE(Q_ensemble_mean, Q_known_val)
nse = NSE(Q_ensemble_mean, Q_known_val)
likelyhood_val = pValObj(Q_rain_ensemble, Q_known_val)
print(f"Rain Ensemble: RMSE = {rmse}, NSE = {nse}, P Val = {likelyhood_val}")
pVal_val = return_pval_obj(Q_rain_ensemble,Q_known_val)
print(f"Validation Pval = {pVal_val}")

plotEnsembleResults(Q_rain_ensemble, Q_mod_val, Qknown, dates_formatted, startday, n, catchment_name_print, model_name, title="Validation Results", ci=95,save_fig=f"./Results/Final_Report/sens_test/{catchment_name}_lam_10.svg")
# , save_fig="./Results/Final_Report/error_cal_process/error_cal_process_cotter_single_obj.svg"
# * ensemble of parameter

# get the ensemble array as input:
# ensemble_array, params_mean_df = getEnsembleArray(model_name, all_params, params_label)

# Q_mod_ensemble = curr_model_ensemble(Rain_input[startday:startday+n], PET_daily, best_params, ensemble_array, n, Area)
# ensemble_std = Q_mod_ensemble.std(axis=1)
# ensemble_mean = Q_mod_ensemble.mean(axis=1)
# top = np.percentile(Q_mod_ensemble, 90, axis=1)
# bot = np.percentile(Q_mod_ensemble, 10, axis=1)
# Q_ensemble_mean = np.expand_dims(ensemble_mean, axis=1)
# rmse = RMSE(Q_ensemble_mean, Q_known_val)
# nse = NSE(Q_ensemble_mean, Q_known_val)
# print("Ensemble Mean: RMSE=", rmse, ", NSE=", nse)
# plotEnsembleResults(Q_mod_ensemble, Q_mod_val, Qknown, dates_formatted, startday, n, catchment_name_print, model_name, title="Parameters", ci=90)

plt.show()


# * Ensemble verification based on Gabrielle 2006


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
print(f"time avd RMSE: {time_avg_rmse} which should equal: {rhs_eqn}")


# some plots for viualisation
# plt.figure()
# plt.plot(ensk_i, label="ensemble skill")
# plt.plot(mse_i, label="mse")
# plt.plot(ensp_i, label="ensp-spread")
# plt.plot(ensk_i, label="ensk")
# plt.plot(kurt_i, label="kurtosis")
# plt.plot(skew_i, label="skew")
# plt.legend()
plt.show()
