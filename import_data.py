from functions import *
from math import isnan
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
# this file is used to import the initial data and clean it and save it in a format to be read by the models.
# ocnvert the input data inot one csv file with date, Percipitation, Qknown, PET

catchment_name = "collie_river"
# Info about the data:
# Cathment values
Area = 672e6  # m2
data_start_date = [1963, 1, 1]
data_end_date = [2019, 2, 28]
folder_name = "./data/"+catchment_name+"/"
Qbos_filename = "612034_daily_ts_formatted.csv"
P_filename = "IDCJAC0009_009738_1800_Data.csv"
P_file_cleandata_idx = 310
PET_filename = "Montly_PET_collie_river.csv"

start_string = str(data_start_date[2])+"/"+str(data_start_date[1])+"/"+str(data_start_date[0])
end_string = str(data_end_date[2])+"/"+str(data_end_date[1])+"/"+str(data_end_date[0])
startdate_formatted = dt.datetime.strptime(start_string, "%d/%m/%Y").date()
enddate_formatted = dt.datetime.strptime(end_string, "%d/%m/%Y").date()
# Streamflow
flow_df = pd.read_csv(folder_name+Qbos_filename)
flow = flow_df[["Date", "Flow (ML)"]].to_numpy()
Qknown = flow[:, 1]/(24*3600)*1e3

# Percipitation
rain_df = pd.read_csv(folder_name+P_filename)
rain_df.drop(rain_df.index[0:P_file_cleandata_idx], inplace=True)
rain = rain_df[["Year", "Month", "Day",
                "Rainfall amount (millimetres)"]].to_numpy()
# Monthly evapoutranpiration
PET_df = pd.read_csv(folder_name+PET_filename)
PET = PET_df["PET(mm)"].to_numpy()

# We only have stream flow or rainfall for a certain amount of time compared to the rainfall data
# Therefore we get the rainfall data for a matching time
rainMathing = getPartialRain(rain, data_start_date, data_end_date)
# and get flow data for a mathcing time if rainfall is shorter.
flowMatching = getPartialFlow(flow, startdate_formatted, enddate_formatted)
# There are sometimes NAN values in our inputs so we need to delete them
idx_nan = []
for i in range(rainMathing.shape[0]):
    if isnan(rainMathing[i, 3]):
        idx_nan = np.append(idx_nan, i)

idx_nan = np.transpose(np.expand_dims(idx_nan, axis=0))
print(idx_nan.shape)
#! The rain data and dates for the period 20/06/1965 to 28/02/2019 with nan values deleted
Rain_input = np.delete(rainMathing, idx_nan.astype(int), axis=0)
#! The streamflow data from 20/06/1965 to 28/02/2019
Q_input = np.delete(flowMatching, idx_nan.astype(int), axis=0)

print(len(rainMathing))
print(len(flowMatching))
# * Create outputs nececasry bt looping through each day
dates = []  # convert the dates into a easy to read format
year_start_idx = pd.DataFrame(columns=["Year", "Month", "Day", "Index"])  # save the indexes of each yeat start
PET_daily_converted = np.zeros([len(Rain_input), 1])  # convert monthly PET values into daily ones

for i in range(len(Rain_input)):
    # finding indexes of Start of years
    if Rain_input[i, 1] == 1 and Rain_input[i, 2] == 1:
        save_vals = np.append(Rain_input[i, 0:3], i)
        year_start_idx.loc[len(year_start_idx.index)] = save_vals
    # creating the dates
    curr_date = str(int(Rain_input[i, 0])) + "/"+str(int(Rain_input[i, 1]))+"/"+str(int(Rain_input[i, 2]))
    dates.append(curr_date)
    # converting monthly PETs into a value for each month
    PET_daily_converted[i] = PET[int(Rain_input[i, 1] - 1)]/1000/(30.5*24*3600)*Area

dates = np.array(dates)
dates_formatted = [dt.datetime.strptime(date, "%Y/%m/%d").date() for date in dates]

# * Plot the data to ensure it is okay
#total_len = 19350
Rtot = Rain_input[:, 3]/1000*Area/(24*3600)  # convert from mm/day to m3/s (also mutiply by catchment area)
Qknown = np.expand_dims(Q_input/(24*3600)*1e3, axis=1)  # convert from ML/day to m3/s

output_df = pd.DataFrame()

output_df["Year"] = Rain_input[:, 0]
output_df["Month"] = Rain_input[:, 1]
output_df["Day"] = Rain_input[:, 2]
output_df["Date"] = dates_formatted
output_df["Percipitation (m3/s)"] = Rtot
output_df["Obs Streamflow (m3/s)"] = Qknown
output_df["PET (m3/s)"] = PET_daily_converted

save = 1
if save == 1:
    input("Areyousure:")
    output_df.to_csv(folder_name+catchment_name+"_inputs.csv", index=False)
    year_start_idx.to_csv(folder_name+"year_start_idx_"+catchment_name+".csv", index=False)  # saving the values of the year start indexes


print(output_df)
plt.plot(dates_formatted, Rtot)
plt.plot(dates_formatted, Qknown)
plt.show()
