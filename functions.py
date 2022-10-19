from math import isnan
import profile
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


def getPartialRain(rain, startDate, endDate):
    # rain = vector of rain data
    # startDate = [Y,M,D] of first day including
    # endDate = [Y,M,D] of last day including
    start_idx = 0
    end_idx = 0
    for i in range(rain.size):
        if rain[i, 0] == startDate[0] and rain[i, 1] == startDate[1] and rain[i, 2] == startDate[2]:
            start_idx = i
        if rain[i, 0] == endDate[0] and rain[i, 1] == endDate[1] and rain[i, 2] == endDate[2]:
            end_idx = i+1
            break

    return rain[start_idx:end_idx]


def getPartialFlow(flow, startDate, endDate):
    # flow is:
    # dd/mm/yyyy
    # flow
    dates_formatted = [dt.datetime.strptime(date, "%d/%m/%Y").date() for date in flow[:, 0]]
    for i in range(len(flow)):
        if dates_formatted[i] == startDate:
            start_idx = i
        if dates_formatted[i] == endDate:
            end_idx = i+1
            break
    return flow[start_idx:end_idx, 1]


def dataYearStartIdx(rain, fileloc):
    # expects rain as [Y,m,d,rain]
    year_start_idx = pd.DataFrame(columns=["Year", "Month", "Day", "Index"])
    for i in range(len(rain)):
        if rain[i, 1] == 1 and rain[i, 2] == 1:
            save_vals = np.append(rain[i, 0:3], i)
            year_start_idx.loc[len(year_start_idx.index)] = save_vals
    year_start_idx.to_csv(fileloc, index=False)
    return


def RMSE(Qmod, Qknown):
    return np.sqrt(np.mean(np.square(Qmod-Qknown)))


def NSE(Qmod, Qknown):
    return (1-(np.sum(np.square(Qmod-Qknown))/np.sum(np.square(Qknown-np.mean(Qknown)))))


def NSE_cal(Qmod, Qknown):
    return ((np.sum(np.square(Qmod-Qknown))/np.sum(np.square(Qknown-np.mean(Qknown)))))


def pValObj_old(Qensemble, Qknown):
    # percentage of values in the 95%CI
    top = np.expand_dims(np.percentile(Qensemble, 95, axis=1), axis=1)
    bot = np.expand_dims(np.percentile(Qensemble, 5, axis=1), axis=1)

    bot_count = np.greater_equal(Qknown, bot)
    top_count = np.less_equal(Qknown, top)
    tot_count = np.sum(np.multiply(bot_count, top_count))

    percent_val = tot_count/len(Qknown)*100
    return percent_val


def pValObj(Qensemble, Qknown):
    # percentage of values in the 95%CI
    top = np.expand_dims(np.percentile(Qensemble, 95, axis=1), axis=1)
    bot = np.expand_dims(np.percentile(Qensemble, 5, axis=1), axis=1)

    bot_count = np.greater_equal(Qknown, bot)
    top_count = np.less_equal(Qknown, top)
    tot_count = np.sum(np.multiply(bot_count, top_count))

    percent_val = tot_count/len(Qknown)*100
    return percent_val


def pso_rmse(params, *args):
    Rain, PET, n, area, Qknown, model = args
    Qmod = model(Rain, PET, params, n, area)
    objfunc = RMSE(Qmod, Qknown)
    return objfunc


def pso_nse(params, *args):
    Rain, PET, n, area, Qknown, model = args
    Qmod = model(Rain, PET, params, n, area)
    objfunc = NSE_cal(Qmod, Qknown)
    return objfunc


def check_pval(Qensemble, Qknown):
    # reutrn the value of PVal and NSE without any waitage applied
    pval_objfunc = abs(90-pValObj(Qensemble, Qknown))
    ensemble_mean = Qensemble.mean(axis=1)
    Q_ensemble_mean = np.expand_dims(ensemble_mean, axis=1)
    nse = RMSE(Q_ensemble_mean, Qknown)
    obj1 = pval_objfunc/10
    obj2 = nse

    return obj1, obj2

def return_pval_obj(Qensemble,Qknown):
    pval_objfunc = abs(90-pValObj(Qensemble, Qknown))
    ensemble_mean = Qensemble.mean(axis=1)
    Q_ensemble_mean = np.expand_dims(ensemble_mean, axis=1)
    nse = RMSE(Q_ensemble_mean, Qknown)
    objfunc = 0.5*pval_objfunc/10 + 0.5*nse
    return objfunc


def pso_pval(params, *args):
    Rain, PET, n, area, Qknown, model = args
    Qensemble = model(Rain, PET, params, n, area)
    obj1 = abs(90-pValObj(Qensemble, Qknown))
    ensemble_mean = Qensemble.mean(axis=1)
    Q_ensemble_mean = np.expand_dims(ensemble_mean, axis=1)
    obj2 = RMSE(Q_ensemble_mean, Qknown)
    objfunc = 0.5*obj1/10 + 0.5*obj2
    return objfunc


def calibrate_likelyhood(folder_loc, date_saved, model_name, catchment_name, print_vals=True):
    dates_formatted, Rain_input, Qknown, PET_daily, catchment_info = getCatchmentInputs(catchment_name)
    Area = float(catchment_info[0])
    startday = int(catchment_info[2])
    n = int(catchment_info[3])
    if model_name == "HYMOD":
        # * HYMOD
        params_label_calibrate = ["Cmax", "Bexp", "Alpha", "Ks", "Kq", "Obj_Func"]
        lb = [1, 0.0001, 0.01, 0.001, 0.00001]
        ub = [10000, 1, 1, 1, 0.1]
        model_cal = models.HYMOD
    elif model_name == "HBV":
        # * HBV
        params_label_calibrate = ["lam", "Smax", "b", "alpha", "Perc", "beta", "gamma", "S2max", "k1", "k2", "sigma", "alpah_correl", "Obj_Func"]
        lb = [0.00001, 0.1*Area, 0.0001, 0.1, 0.000001*Area / (24*3600), 0.00001, 0.1, 0.1*0.2*Area, 0.0001/(24*3600), 5*Area/(24*3600), 0.1, 0.1]
        ub = [10, 50*Area, 200, 0.95, 30*Area/(24*3600), 1.5, 50, 50*0.2*Area, 10/(24*3600), 350000*Area/(24*3600), 10, 0.9]
        model_cal = models.HBVRainEnsemble
    if print_vals == True:
        print("Calibration bound is")
        bounds_df = pd.DataFrame(columns=params_label_calibrate[0:-1])
        bounds_df.loc[len(bounds_df)] = np.transpose(lb)
        bounds_df.loc[len(bounds_df)] = np.transpose(ub)
        print(bounds_df)
        print("-------------")

    args_in = (Rain_input[startday:startday+n], PET_daily[startday:startday+n], n, Area, Qknown[startday:startday+n], model_cal)

    all_params_df = pd.DataFrame(columns=params_label_calibrate)
    filename = "all_params_pVal_"+model_name+"_"+catchment_name+"_"+date_saved+".csv"
    
    for i in range(5):
        cal_params, fopt = pso(pso_pval, lb, ub, args=args_in, debug=False)
        save_vals = np.append(cal_params, fopt)
        all_params_df.loc[len(all_params_df.index)] = save_vals
        print(all_params_df)
        all_params_df.to_csv(folder_loc+filename, index=False)
    all_params_df.to_csv(folder_loc+filename, index=False)
    return


def calibrate_model(folder_loc, date_saved, model_name, catchment_name, args_in, lb, ub, params_label):
    n_runs = 5
    all_params_df = pd.DataFrame(columns=params_label)
    rmse_params_df = pd.DataFrame(columns=params_label)
    filename = "params_rmse_"+model_name+"_"+catchment_name+"_"+date_saved+".csv"
    for i in range(n_runs):
        cal_params, fopt = pso(pso_rmse, lb, ub, args=args_in, debug=False)
        save_vals = np.append(cal_params, fopt)
        rmse_params_df.loc[len(rmse_params_df.index)] = save_vals
        all_params_df.loc[len(all_params_df.index)] = save_vals
        print(all_params_df)
        rmse_params_df = rmse_params_df.sort_values(by="Obj_Func")
        rmse_params_df.to_csv(folder_loc+filename, index=False)

    nse2_params_df = pd.DataFrame(columns=params_label)
    filename = "params_nse_"+model_name+"_"+catchment_name+"_"+date_saved+".csv"
    for i in range(n_runs):
        cal_params, fopt = pso(pso_nse, lb, ub, args=args_in, debug=False)
        save_vals = np.append(cal_params, fopt)
        nse2_params_df.loc[len(nse2_params_df.index)] = save_vals
        all_params_df.loc[len(all_params_df.index)] = save_vals
        print(all_params_df)
        nse2_params_df = nse2_params_df.sort_values(by="Obj_Func")
        nse2_params_df.to_csv(folder_loc+filename, index=False)

    filename = "all_params_"+model_name+"_"+catchment_name+"_"+date_saved+".csv"
    all_params_df.to_csv(folder_loc+filename, index=False)
    return


def prepCalModel(catchment_list, model_tocal_list, folder_loc, date_saved):
    for i in range(len(catchment_list)):
        catchment_name = catchment_list[i]  # "macintyre_river"
        dates_formatted, Rain_input, Qknown, PET_daily, catchment_info = getCatchmentInputs(catchment_name)
        Area = float(catchment_info[0])
        catchment_name_print = catchment_info[1][0:-1]

        # import required model parameters
        for j in range(len(model_tocal_list)):
            model_name = model_tocal_list[j]

            startday = int(catchment_info[2])
            n = 50  # int(catchment_info[3])

            if model_name == "HYMOD":
                # * HYMOD
                params_label_calibrate = ["Cmax", "Bexp", "Alpha", "Ks", "Kq", "Obj_Func"]
                lb = [1, 0.0001, 0.01, 0.001, 0.00001]
                ub = [10000, 1, 1, 1, 0.1]
                model_cal = models.HYMOD
            elif model_name == "HBV":
                # * HBV
                params_label_calibrate = ["lam", "Smax", "b", "alpha", "Perc", "beta", "gamma", "S2max", "k1", "k2", "Obj_Func"]
                lb = [0.00001, 0.1*Area, 0.01, 0.1, 0.0001*Area / (24*3600), 0.00001, 0.1, 0.1*0.2*Area, 0.0001/(24*3600), 0.05*Area/(24*3600)]
                ub = [10, 50*Area, 200, 0.95, 30*Area/(24*3600), 1.5, 50, 50*0.2*Area, 10/(24*3600), 350*Area/(24*3600)]
                model_cal = models.HBV
            print("Calibration bound is")
            bounds_df = pd.DataFrame(columns=params_label_calibrate[0:-1])
            bounds_df.loc[len(bounds_df)] = np.transpose(lb)
            bounds_df.loc[len(bounds_df)] = np.transpose(ub)
            print(bounds_df)
            print("-------------")
            args = (Rain_input[startday:startday+n], PET_daily[startday:startday+n], n, Area, Qknown[startday:startday+n], model_cal)
            calibrate_model(folder_loc, date_saved, model_name, catchment_name, args, lb, ub, params_label_calibrate)
    return


def read_params(filename):
    df = pd.read_csv(filename)
    params = df[["Values"]].to_numpy()
    return params


def plot_streamflow(Qknown, dates, start_day, n, Qmod=[-1], save="None", Rain=[-1]):
    plt.rc('font', size=13)
    if Rain[0] != -1:
        plt.plot(dates[start_day:start_day+n], Rain, "k", label="Rain")
    plt.plot(dates[start_day:start_day+n], Qknown[start_day:start_day+n], "b", label="$Q_{observed}$")
    if Qmod[0] != -1:
        plt.plot(dates[start_day:start_day+n], Qmod, "r", label="$Q_{predicted}$")
    plt.legend()
    plt.ylabel("Flow ($m^3$/s)")
    plt.xlabel("Date")
    plt.ylim(bottom=0)

    if save != "None":
        plt.savefig(save)


def getCatchmentInputs(catchment_name):

    input_loc = "./data/"+catchment_name+"/"
    input_df = pd.read_csv(input_loc+catchment_name+"_inputs.csv")
    input_data = input_df[["Date", "Percipitation (m3/s)", "Obs Streamflow (m3/s)", "PET (m3/s)"]].to_numpy()
    dates_formatted = [dt.datetime.strptime(date, "%Y-%m-%d").date() for date in input_data[:, 0]]
    with open(input_loc+catchment_name+"_info.txt") as f:
        catchment_info = f.readlines()

    Pinput = input_data[:, 1]
    Qobs = input_data[:, 2]
    Qobs = np.expand_dims(Qobs, axis=1)
    PET_daily = input_data[:, 3]
    return dates_formatted, Pinput, Qobs, PET_daily, catchment_info


def getModelData(model_name, catchment_name, alt_loc=-1, best_idx=0, print_vals=False):
    if model_name == "HBV":
        end_idx = 10
        curr_model = models.HBV
        curr_ensemble_model = models.HBV_ensemble
    elif model_name == "HYMOD":
        end_idx = 5
        curr_model = models.HYMOD
        curr_ensemble_model = models.HYMOD_ensemble
    elif model_name == "HBV_likelyhood":
        end_idx = 13
        curr_model = models.HBVRainEnsemble
        curr_ensemble_model = models.HBVRainEnsemble
    else:
        print("worng model")
        return
    if alt_loc == -1:
        input_loc = "./parameters/Calibrated_choosen/"+model_name+"_params"+"_"+catchment_name
        best_params_df = pd.read_csv(input_loc+"_choosen.csv")
        all_params_df = pd.read_csv(input_loc+"_all.csv")
    else:
        all_params_df = pd.read_csv(alt_loc)
        best_params_df = pd.DataFrame(columns=all_params_df.columns)
        best_params_df.loc[len(best_params_df)] = all_params_df.iloc[best_idx, :]

    best_params = best_params_df.to_numpy()
    best_params = best_params.transpose()
    all_params = all_params_df.iloc[:, 0:end_idx].to_numpy()
    params_label = all_params_df.columns[0:end_idx].values
    if print_vals == True:
        print("------")
        print(all_params_df)
        print("------")
        print("The Choosen parameters is:")
        print(best_params_df)
        print("------")

    return curr_model, curr_ensemble_model, best_params, all_params, all_params_df, params_label


def getEnsembleArray(model_name, all_params, params_label):
    params_mean = all_params.mean(axis=0)
    params_mean = np.expand_dims(params_mean, axis=0)
    params_std = np.expand_dims(all_params.std(axis=0), axis=0)
    params_mean_df = pd.DataFrame(params_mean, columns=params_label)
    params_mean_df.loc[1, :] = params_std
    print("The mean & std of the parameters is:")
    print(params_mean_df)
    print("------")

    if model_name == "HBV":
        choose_ensemble = [0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
        # i=4
        # choose_ensemble[i]=1
        choose_ensemble = np.expand_dims(choose_ensemble, axis=0)
        ensemble_array = np.array([params_mean, params_std, choose_ensemble])
        ensemble_array = np.transpose(np.squeeze(ensemble_array, axis=1))
        # manually adjusting some ensemble values

        # ensemble_array[0,1]=0.05 #lam both mean and std
        # ensemble_array[1,1]=6e8 #Smax, std
        # ensemble_array[2, [0, 1]] = [3, 4]  # b both mean and std
        # ensemble_array[4,[0,1]]=[750,500] #b both mean and std
        # # ensemble_array[7,1]=6e8
        # ensemble_array[8,1]=1e-5
        # ensemble_array[9,1]=20000
    elif model_name == "HYMOD":
        choose_ensemble = [0, 0, 0, 0, 0]
        i = 0
        choose_ensemble[i] = 1
        choose_ensemble = np.expand_dims(choose_ensemble, axis=0)
        ensemble_array = np.array([params_mean, params_std, choose_ensemble])
        ensemble_array = np.transpose(np.squeeze(ensemble_array, axis=1))
        # print(ensemble_array)
        # manually adjusting some ensemble values
        # ensemble_array[0, 1] = 250
    return ensemble_array, params_mean_df


def printNormalResults(Qmod, Qknown, optional_starter=":"):
    rmse = RMSE(Qmod, Qknown)
    nse = NSE(Qmod, Qknown)
    
    print(f"{optional_starter} RMSE={rmse}, NSE = {nse}")


def plotResultsPercipBar(Q_mod_val, Qknown, Rain_input, dates_formatted, startday, n, catchment_name, model_name, title=".", save_fig=-1):
    plt.rc('font', size=13)
    fig1 = plt.figure(figsize=(10, 5.5))
    plt.tight_layout()
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
    fig1.suptitle("Catchment: "+catchment_name+", Model: "+model_name, fontsize=15)
    # top middle plot of the model prediciton
    ax1 = plt.subplot(gs[1])
    ax1.plot(dates_formatted[startday:startday+n], Qknown[startday:startday+n], "b", label="Observed")
    ax1.plot(dates_formatted[startday:startday+n], Q_mod_val,  "r", label="Predicted")
    ax1.set_ylabel("Streamflow ($m^3$/s)")
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(dates_formatted[startday], dates_formatted[startday+n])
    # ax1.tick_params('x', labelsize=False)
    ax1.set_xlabel("Time (Days)")
    ax1.legend()
    # top bar graph of percipitation
    ax2 = plt.subplot(gs[0])
    ax2.bar(dates_formatted[startday:startday+n], Rain_input[startday:startday+n])
    ax2.tick_params('x', labelbottom=False)
    ax2.set_ylabel("Precipitation ($m^3$/s)")
    ax2.set_xlim(dates_formatted[startday], dates_formatted[startday+n])
    if save_fig != -1:
        fig1.savefig(save_fig, format='svg', bbox_inches='tight')
    return


def plotResiduals(Q_mod_val, Qknown, Rain_input, dates_formatted, startday, n, catchment_name, model_name, save_fig=-1):
    residuals = Qknown[startday:startday+n]-Q_mod_val
    plt.rc('font', size=10)
    fig1 = plt.figure(figsize=(10, 5.5))
    fig1.suptitle("Catchment: "+catchment_name+", Model: "+model_name, fontsize=15)
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 1])
    # middle plot of the model prediciton
    ax1 = plt.subplot(gs[1])
    ax1.plot(dates_formatted[startday:startday+n], Qknown[startday:startday+n], "b", label="Observed")
    ax1.plot(dates_formatted[startday:startday+n], Q_mod_val,  "r", label="Predicted")
    ax1.set_ylabel("Streamflow ($m^3$/s)")
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(dates_formatted[startday], dates_formatted[startday+n])
    ax1.tick_params('x', labelsize=False)
    ax1.legend()
    # top bar graph of percipitation
    ax2 = plt.subplot(gs[0])
    ax2.bar(dates_formatted[startday:startday+n], Rain_input[startday:startday+n])
    ax2.tick_params('x', labelbottom=False)
    ax2.set_ylabel("Precipitation (mm/d)")
    ax2.set_xlim(dates_formatted[startday], dates_formatted[startday+n])
    # bottom plot of residuals
    ax3 = plt.subplot(gs[2])
    ax3.plot(dates_formatted[startday:startday+n], residuals, 'k')
    ax3.tick_params('x', labelbottom=13)
    ax3.set_ylabel("Residuals ($m^3$/s)")
    ax3.set_xlim(dates_formatted[startday], dates_formatted[startday+n])
    ax3.set_xlabel("Time (Days)")
    if save_fig != -1:
        fig1.savefig(save_fig, format='svg', bbox_inches='tight')
    return


def plotEnsembleResults(Q_mod_ensemble, Q_mod_val, Qknown, dates_formatted, startday, n, catchment_name, model_name, ci=90, title=".", save_fig=-1):
    ensemble_mean = Q_mod_ensemble.mean(axis=1)
    top = np.percentile(Q_mod_ensemble, ci, axis=1)
    bot = np.percentile(Q_mod_ensemble, 100-ci, axis=1)

    plt.rc('font', size=13)
    fig2 = plt.figure(figsize=(10, 5.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    fig2.suptitle("Catchment: "+catchment_name+", Model: "+model_name, fontsize=15)
    ax1 = plt.subplot(gs[0])
    ax1.fill_between(dates_formatted[startday:startday+n], top, bot, color='lightgrey')
    ax1.plot(dates_formatted[startday:startday+n], top, 'k', label=f"{ci}th percentile", linewidth=0.5)
    ax1.plot(dates_formatted[startday:startday+n], bot, 'k', label=f"{100-ci}th percentile", linewidth=0.5)
    ax1.plot(dates_formatted[startday:startday+n], Qknown[startday:startday+n], '.', label="Observed", color='black')
    # ax1.plot(dates_formatted[startday:startday+n], ensemble_mean, label="Ensemble Prediction", color='red')
    ax1.set_ylabel("Streamflow ($m^3$/s)")
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(dates_formatted[startday], dates_formatted[startday+n])
    ax1.tick_params('x', labelsize=False)
    ax1.legend(loc='upper left')
    ax2 = plt.subplot(gs[1])
    ax2.plot(dates_formatted[startday:startday+n], Q_mod_val, '--', label="Calibrated Prediction", color='blue', linewidth=0.5)
    ax2.plot(dates_formatted[startday:startday+n], ensemble_mean, label="Ensemble Prediction", color='red')
    ax2.plot(dates_formatted[startday:startday+n], Qknown[startday:startday+n], '.', label="Observed", color='black')
    ax2.tick_params('x', labelbottom=13)
    ax2.set_xlabel("Time (Days)")
    ax2.set_ylabel("Streamflow ($m^3$/s)")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper left')
    ax2.set_xlim(dates_formatted[startday], dates_formatted[startday+n])
    if save_fig != -1:
        fig2.savefig(save_fig, format='svg', bbox_inches='tight')
    return


def changeFileNames(src, dest, date, catch_list, model_list):
    for currname in catch_list:
        for currmodel in model_list:
            old_file_name = f"{src}/all_params_{currmodel}_{currname}_{date}.csv"
            dest = f"{dest}/{currmodel}_params_{currname}_all.csv"
            os.rename(old_file_name, dest)
    return


def saveBestParam(all_params_df, best_idx, model_name, catchment_name):
    best_params_df = all_params_df.iloc[best_idx, :]
    dest_name = f"./parameters/Calibrated_choosen/{model_name}_params_{catchment_name}_choosen.csv"
    best_params_df.to_csv(dest_name, index=False)
    return


def getCorrel(input_data, lag=1):

    Q_df = pd.DataFrame(input_data)
    Q_df_shifted = pd.concat([Q_df, Q_df.shift(1), Q_df.shift(2)], axis=1)
    Q_df_shifted.columns = ['t', 't+1', 't+2']
    Q_corr = Q_df_shifted.corr()["t"].to_numpy()
    if lag == 1:
        return Q_corr[1]
    elif lag == 2:
        return Q_corr[2]
    else:
        print("only lag 1 or 2 accepted")
        return "Error"


print("Run the main.py dumbass")
