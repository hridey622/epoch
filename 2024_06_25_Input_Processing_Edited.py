import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#import scipy

from scipy import interpolate

import time
start = time.time()

# hd5 file description

# cycle_stats: 
 #   Column              Non-Null Count  Dtype  
#---  ------              --------------  -----  
 #0   cycle_number        348 non-null    int64  
 #1   charge_energy       338 non-null    float64
 #2   charge_capacity     338 non-null    float64
 #3   discharge_energy    347 non-null    float64
 #4   discharge_capacity  347 non-null    float64

# raw_data: 
 #   Column         Non-Null Count  Dtype  
#---  ------         --------------  -----  
 #0   cycle_number   71184 non-null  int64  
 #1   file_number    71184 non-null  int64  
 #2   test_time      71184 non-null  float64
 #3   state          71184 non-null  object 
 #4   current        71184 non-null  float64
 #5   voltage        71184 non-null  float64
 #6   step_index     71184 non-null  int64  
 #7   method         71184 non-null  object 
 #8   substep_index  71184 non-null  int64  


# files within the folder and providing input file location
import os 

folder = 'Other - Incomplete data/'
#folder = 'Refined - Incomplete data/'

input_path = '/ANL/Datasets/Input/' + folder
output_path = '/ANL/Datasets/Output/' + folder

dir_list = os.listdir(input_path) 
print("Files and directories in '", input_path, "' :")  
   
# print the list
print(dir_list)
print(dir_list.count)

cell_df = pd.DataFrame(columns=("cell ID", "V_mean", "V_std")) #, "voltage_mean", "voltage_std", "voltage_IQR", "voltage_max", "voltage_min", "dis_capacity_nominal"))
l = len(cell_df.index)


for cell_ID in dir_list:
    cell_ID = cell_ID.replace('.h5', '')
    print(cell_ID)

    file_name = input_path + cell_ID + ".h5"

    with h5py.File(file_name, 'r') as f:
        data1_set = f['cycle_stats/table']
        data2_set = f['raw_data/table']
        data1 = data1_set[:1000]
        data2 = data2_set[:100000]

    cycle_df = pd.read_hdf(file_name, key="cycle_stats")
    rawdata_df = pd.read_hdf(file_name, key="raw_data")

    cycleID = rawdata_df["cycle_number"].unique().tolist()
    stateID = rawdata_df["state"].unique().tolist()

    outliers1_full = pd.DataFrame()
    m0cycle_df = pd.DataFrame()
    m1cycle_df = pd.DataFrame()
    m2cycle_df = pd.DataFrame()
    m3cycle_df = pd.DataFrame()

    if cycle_df["cycle_number"].count() > 100: #only considering cells with cycle life > 100

        # estimating capacity and energy for each cycle and error wrt to "cycle_stats" file; error needs to analyzed and corrected, as error seems to be high 
        rawdata_df["dis_Q"] = 0
        rawdata_df["ch_Q"] = 0
        rawdata_df["dis_E"] = 0
        rawdata_df["ch_E"] = 0

        rawdata_df["dis_Q"] = rawdata_df["dis_Q"].astype(float)
        rawdata_df["ch_Q"] = rawdata_df["ch_Q"].astype(float)
        rawdata_df["dis_E"] = rawdata_df["dis_E"].astype(float)
        rawdata_df["ch_E"] = rawdata_df["ch_E"].astype(float)

        aa = bb = cc = dd = 0

        for x in cycleID:
   
            for row in rawdata_df.itertuples():   
                if rawdata_df.loc[row.Index, "cycle_number"] == x:
                    if rawdata_df.loc[row.Index, "state"] == "charging":
                        rawdata_df.loc[row.Index, "ch_Q"] = aa + (rawdata_df.loc[row.Index, "current"] + rawdata_df.loc[(row.Index -1), "current"])/2 * (rawdata_df.loc[row.Index, "test_time"] - rawdata_df.loc[(row.Index -1), "test_time"])/3600
                        rawdata_df.loc[row.Index, "ch_E"] = bb + (rawdata_df.loc[row.Index, "current"] + rawdata_df.loc[(row.Index -1), "current"])/2 * (rawdata_df.loc[row.Index, "test_time"] - rawdata_df.loc[(row.Index -1), "test_time"])/3600 * (rawdata_df.loc[row.Index, "voltage"] + rawdata_df.loc[(row.Index -1), "voltage"])/2
                        aa = rawdata_df.loc[row.Index, "ch_Q"]
                        bb = rawdata_df.loc[row.Index, "ch_E"]
                    if rawdata_df.loc[row.Index, "state"] == "discharging":
                        rawdata_df.loc[row.Index, "dis_Q"] = cc - (rawdata_df.loc[row.Index, "current"] + rawdata_df.loc[(row.Index -1), "current"])/2 * (rawdata_df.loc[row.Index, "test_time"] - rawdata_df.loc[(row.Index -1), "test_time"])/3600
                        rawdata_df.loc[row.Index, "dis_E"] = dd - (rawdata_df.loc[row.Index, "current"] + rawdata_df.loc[(row.Index -1), "current"])/2 * (rawdata_df.loc[row.Index, "test_time"] - rawdata_df.loc[(row.Index -1), "test_time"])/3600 * (rawdata_df.loc[row.Index, "voltage"] + rawdata_df.loc[(row.Index -1), "voltage"])/2
                        cc = rawdata_df.loc[row.Index, "dis_Q"]
                        dd = rawdata_df.loc[row.Index, "dis_E"]
               
            cycle_df.loc[x, "dis_Q1"] = cc
            cycle_df.loc[x, "ch_Q1"] = aa
            cycle_df.loc[x, "dis_E1"] = dd
            cycle_df.loc[x, "ch_E1"] = bb   
  
            if x == 0:
                cycle_df.loc[x, "cumm_dis_Q1"] = cc
                cycle_df.loc[x, "cumm_dis_E1"] = dd

            if x > 0:
                cycle_df.loc[x, "cumm_dis_Q1"] = cycle_df.loc[(x-1), "cumm_dis_Q1"] + cc
                cycle_df.loc[x, "cumm_dis_E1"] = cycle_df.loc[(x-1), "cumm_dis_E1"] + dd

            aa = bb = cc = dd = 0

        cycle_df["d_Q_error%"] = (cycle_df["dis_Q1"] + cycle_df["discharge_capacity"])/(-cycle_df["discharge_capacity"])*100
        cycle_df["c_Q_error%"] = (cycle_df["ch_Q1"] - cycle_df["charge_capacity"])/cycle_df["charge_capacity"]*100
        cycle_df["d_E_error%"] = (cycle_df["dis_E1"] + cycle_df["discharge_energy"])/(-cycle_df["discharge_energy"])*100
        cycle_df["c_E_error%"] = (cycle_df["ch_E1"] - cycle_df["charge_energy"])/cycle_df["charge_energy"]*100

        cycle_df["Q_Eff"] = cycle_df["dis_Q1"] / cycle_df["ch_Q1"] * 100
        cycle_df["E_Eff"] = cycle_df["dis_E1"] / cycle_df["ch_E1"] * 100

        # statistics for each cycle - voltage, capacity degradation (slope)
        for x in cycleID:
            subset_rawdata_df = rawdata_df[(rawdata_df["cycle_number"] == x)] 
            cycle_df.loc[x, "V_mean"] = subset_rawdata_df["voltage"].mean()
            cycle_df.loc[x, "V_std"] = subset_rawdata_df["voltage"].std()
            cycle_df.loc[x, "V_IQR"] = subset_rawdata_df["voltage"].quantile(0.75) - subset_rawdata_df["voltage"].quantile(0.25)
            cycle_df.loc[x, "V_max"] = subset_rawdata_df["voltage"].max()
            cycle_df.loc[x, "V_min"] = subset_rawdata_df["voltage"].min()
            cycle_df.loc[x, "V_count"] = subset_rawdata_df["voltage"].count()
        
            d_subset_rawdata_df = subset_rawdata_df[(subset_rawdata_df["state"] == "discharging")]
            cycle_df.loc[x, "I_dis_mean"] = -d_subset_rawdata_df["current"].mean()
  
            # slope calc for cap degradation using linear curve fitting (calling it m1)
            if x > 0 and x < 6:
                cycle_df.loc[x, "Q_deg_slope%"] = (cycle_df.loc[x, "dis_Q1"] - cycle_df.loc[(x-1), "dis_Q1"]) / cycle_df.loc[x, "dis_Q1"]*100 
            if x >= 6:
                m1cycle_df = cycle_df[cycle_df["cycle_number"].isin([(x-5), (x-4), (x-3), (x-2), (x-1), x, (x+1), (x+2), (x+3), (x+4), (x+5)])]

                m1_Q1 = m1cycle_df["dis_Q1"].quantile(0.25)
                m1_Q3 = m1cycle_df["dis_Q1"].quantile(0.75)
                m1_IQR = m1_Q3 - m1_Q1
                threshold = 1.5
                outliers1 = m1cycle_df[(m1cycle_df['dis_Q1'] < m1_Q1 - threshold * m1_IQR) | (m1cycle_df['dis_Q1'] > m1_Q3 + threshold * m1_IQR)]
                m1cycle_df = m1cycle_df.drop(outliers1.index)

                outliers1_full = pd.concat([outliers1_full, outliers1]) #compiling outliers1 into a df

                # Splitting variables for regression
                X1 = m1cycle_df["cycle_number"].values  # independent
                X1_shaped = X1.reshape(-1, 1)
                y1 = m1cycle_df["dis_Q1"].values  # dependent

                # Train linear regression model on whole dataset
                lr1 = LinearRegression()
                lr1.fit(X1_shaped, y1)

                cycle_df.loc[x, "Q_deg_slope%"] = lr1.coef_

        # outliers treatment (calling it m3) on I-discharge and capacity (outliers1_full); Delta-Q interpolation 
        cycle_df = cycle_df.iloc[1: , :]
        I_mean = cycle_df["I_dis_mean"].mode()[0]
        outliers2 = cycle_df[(cycle_df['I_dis_mean'] < I_mean*0.85) | (cycle_df['I_dis_mean'] > I_mean*1.15)]

        outliers1_full.drop_duplicates(subset=['cycle_number'], keep='first', inplace=True)       
        
        outliers = pd.concat([outliers2, outliers1_full])
        outliers.drop_duplicates(subset=['cycle_number'], keep='first', inplace=True)

        m3cycle_df = cycle_df.drop(outliers.index)

        #print(m3_Q1, m3_Q3, m3_IQR)

        cycle3ID = m3cycle_df["cycle_number"].unique().tolist()

        # Q-nominal calc based on first 3 cycles and then C-rate calc
        m4cycle_df = m3cycle_df.iloc[1:4,]
        cycle_df["dis_Q_nom"] = m4cycle_df["dis_Q1"].sum() / m4cycle_df["dis_Q1"].count()

        cycle_df["dis_C-rate"] = cycle_df["I_dis_mean"] / cycle_df["dis_Q_nom"]


        # Splitting variables for regression
        X2 = m3cycle_df["cycle_number"].values  # independent
        X2_shaped = X2.reshape(-1, 1)
        y2 = m3cycle_df["dis_Q1"].values  # dependent

        # Train linear regression model on whole dataset (m2)
        lr2 = LinearRegression()
        lr2.fit(X2_shaped, y2)

        # Train polynomial regression model on the whole dataset (m2)
        pr2 = PolynomialFeatures(degree = 4)
        X2_poly = pr2.fit_transform(X2_shaped)
        pr2.fit(X2_poly, y2)
        lr2b = LinearRegression()
        lr2b.fit(X2_poly, y2)

        print(cycle3ID)    

        def closest(cycle3ID, K):
     
            return cycle3ID[min(range(len(cycle3ID)), key = lambda i: abs(cycle3ID[i]-K))]

        m = closest(cycle3ID, 10) # estimating closest cycle to 10
        n = closest(cycle3ID, 25) # estimating closest cycle to 25
        o = closest(cycle3ID, 100) # estimating closest cycle to 100
    
        cycle10_rawdata_df = rawdata_df[(rawdata_df["cycle_number"] == m) & (rawdata_df["state"] == "discharging")] 
        cycle25_rawdata_df = rawdata_df[(rawdata_df["cycle_number"] == n) & (rawdata_df["state"] == "discharging")] 
        cycle100_rawdata_df = rawdata_df[(rawdata_df["cycle_number"] == o) & (rawdata_df["state"] == "discharging")] 

        # delta-Q interpolation and diffference between cycle 10, 25, and 100
        X3 = cycle10_rawdata_df["voltage"].values  # independent
        y3 = cycle10_rawdata_df["dis_Q"].values  # dependent

        X4 = cycle25_rawdata_df["voltage"].values  # independent
        y4 = cycle25_rawdata_df["dis_Q"].values  # dependent

        X5 = cycle100_rawdata_df["voltage"].values  # independent
        y5 = cycle100_rawdata_df["dis_Q"].values  # dependent

        f = interpolate.interp1d(X3, y3, kind='nearest',fill_value="extrapolate")
        
        # cycle 25 vs. 10
        y3i = f(X4) 
        delta_y = y4 - y3i

        # cycle 100 vs. 10
        y3i_2 = f(X5)
        delta_y2 = y5 - y3i_2
       

        # cycle life estimation
        EOL_capacity = 0.8 * cycle_df["dis_Q_nom"].mean()

        for x in cycle3ID:
            if m3cycle_df.loc[x, "dis_Q1"] > EOL_capacity:
                EOL_cycle = m3cycle_df.loc[x, "cycle_number"]
    
        # cell statistics
        l = l + 1
        cell_df.loc[l, "cell ID"] = cell_ID  
        cell_df.loc[l, "V_mean"] = cycle_df["V_mean"].mean()
        cell_df.loc[l, "V_std"] = cycle_df["V_std"].mean()
        cell_df.loc[l, "V_IQR"] = cycle_df["V_IQR"].mean()
        cell_df.loc[l, "V_max"] = cycle_df["V_max"].max()
        cell_df.loc[l, "V_min"] = cycle_df["V_min"].min()
        cell_df.loc[l, "dis_Q_nom"] = cycle_df["dis_Q_nom"].mean()
        cell_df.loc[l, "dis_C-rate_mode"] = cycle_df["dis_C-rate"].mode()[0]
        cell_df.loc[l, "dis_C-rate_max"] = cycle_df["dis_C-rate"].max()
        cell_df.loc[l, "cycles_tested"] = cycle_df["cycle_number"].max()
        cell_df.loc[l, "cycle_life"] = EOL_cycle
        cell_df.loc[l, "cycle_10"] = m
        cell_df.loc[l, "Q_10"] = cycle_df.loc[m, "dis_Q1"] 
        cell_df.loc[l, "Q_deg_10"] = cycle_df.loc[m, "Q_deg_slope%"] 
        cell_df.loc[l, "Q_Eff_10"] = cycle_df.loc[m, "Q_Eff"]
        cell_df.loc[l, "E_Eff_10"] = cycle_df.loc[m, "E_Eff"]   
        cell_df.loc[l, "cumm_Q_10"] = cycle_df.loc[m, "cumm_dis_Q1"] 
        cell_df.loc[l, "cumm_E_10"] = cycle_df.loc[m, "cumm_dis_E1"]

        cell_df.loc[l, "cycle_25"] = n
        cell_df.loc[l, "Q_25"] = cycle_df.loc[n, "dis_Q1"] 
        cell_df.loc[l, "Q_deg_25"] = cycle_df.loc[n, "Q_deg_slope%"]
        cell_df.loc[l, "Q_Eff_25"] = cycle_df.loc[n, "Q_Eff"]
        cell_df.loc[l, "E_Eff_25"] = cycle_df.loc[n, "E_Eff"]   
        cell_df.loc[l, "cumm_Q_25"] = cycle_df.loc[n, "cumm_dis_Q1"] 
        cell_df.loc[l, "cumm_E_25"] = cycle_df.loc[n, "cumm_dis_E1"]

        cell_df.loc[l, "cycle_100"] = o
        cell_df.loc[l, "Q_100"] = cycle_df.loc[o, "dis_Q1"] 
        cell_df.loc[l, "Q_deg_100"] = cycle_df.loc[o, "Q_deg_slope%"]
        cell_df.loc[l, "Q_Eff_100"] = cycle_df.loc[o, "Q_Eff"]
        cell_df.loc[l, "E_Eff_100"] = cycle_df.loc[o, "E_Eff"]   
        cell_df.loc[l, "cumm_Q_100"] = cycle_df.loc[o, "cumm_dis_Q1"] 
        cell_df.loc[l, "cumm_E_100"] = cycle_df.loc[o, "cumm_dis_E1"]

        cell_df.loc[l, "delta-Q25_mean"] = delta_y.mean()
        cell_df.loc[l, "delta-Q25_std"] = delta_y.std()
        cell_df.loc[l, "delta-Q25_IQR"] = np.percentile(delta_y, 75) - np.percentile(delta_y, 25)
        cell_df.loc[l, "delta-Q25_max"] = delta_y.max()
        cell_df.loc[l, "delta-Q25_min"] = delta_y.min()
        cell_df.loc[l, "delta-E25"] = cycle_df.loc[m, "dis_E1"] - cycle_df.loc[n, "dis_E1"]
        cell_df.loc[l, "delta-Q100_mean"] = delta_y2.mean()
        cell_df.loc[l, "delta-Q100_std"] = delta_y2.std()
        cell_df.loc[l, "delta-Q100_IQR"] = np.percentile(delta_y2, 75) - np.percentile(delta_y2, 25)
        cell_df.loc[l, "delta-Q100_max"] = delta_y2.max()
        cell_df.loc[l, "delta-Q100_min"] = delta_y2.min()
        cell_df.loc[l, "delta-E100"] = cycle_df.loc[m, "dis_E1"] - cycle_df.loc[o, "dis_E1"]
    
        cell_df.to_csv(output_path + 'cell.csv') 
        print(cell_df.iloc[0:50, 0:20])
        print(cell_df.iloc[0:50, 20:40])
        print(cell_df.iloc[0:50, 40:60])

        if not os.path.exists(output_path + cell_ID): 
            os.makedirs(output_path + cell_ID) 

        rawdata_df.to_csv(output_path + cell_ID + '/' + cell_ID + '_rawdata.csv')
        cycle_df.to_csv(output_path + cell_ID + '/' + cell_ID + '_cycle.csv')

        # printing and plotting
        plt.figure()
        ax1 = plt.gca()
        cycle_df.plot(kind="scatter", x="cycle_number", y="dis_Q1", color="xkcd:dark grey", ax=ax1)
        m3cycle_df.plot(kind="scatter", x="cycle_number", y="dis_Q1", color="xkcd:light blue", ax=ax1)
        plt.plot(X2, lr2.predict(X2_shaped), color = 'red')
        plt.plot(X2, lr2b.predict(X2_poly), color = 'blue')
        plt.xlabel("Cycle Number")
        plt.ylabel("Discharge Capacity")
        ax1.legend(["Raw Data", "Outliers Removed", "Linear Fit", "Polynomial Fit"])
        plt.title(cell_ID)
        plt.savefig(output_path + cell_ID + '/' + cell_ID + "_Q_cycle.png")
        plt.show(block = False)
        plt.pause(0.5)
        plt.close()   
        plt.clf()

        plt.figure()
        ax2 = plt.gca()
        rawdata_df.plot(kind="scatter", x="dis_Q", y="voltage", color="xkcd:dark grey", label = "Raw Data", ax=ax2)
        plt.xlabel("Capacity")
        plt.ylabel("Voltage")
        plt.title(cell_ID)
        plt.savefig(output_path + cell_ID + '/' + cell_ID + "_Q(V).png")   
        plt.show(block = False)
        plt.pause(0.5)
        plt.close()   
        plt.clf()  

        fig3 = plt.figure()
        ax3 = plt.gca()
        cycle10_rawdata_df.plot(kind="scatter", x="voltage", y="dis_Q", color="xkcd:dark grey", label = "Cycle " + str(m), ax=ax3)
        cycle25_rawdata_df.plot(kind="scatter", x="voltage", y="dis_Q", color="xkcd:dark red", label = "Cycle " + str(n), ax=ax3)
        cycle100_rawdata_df.plot(kind="scatter", x="voltage", y="dis_Q", color="xkcd:dark blue", label = "Cycle " + str(o), ax=ax3)
        plt.xlabel("Voltage")
        plt.ylabel("Capacity")
        plt.title(cell_ID)
        plt.savefig(output_path + cell_ID + '/' + cell_ID + "_Q(V)_cycle_10_25_100.png")   
        plt.show(block = False)
        plt.pause(0.5)
        plt.close()   
        plt.clf() 

        fig4 = plt.figure()
        ax4 = plt.gca()
        plt.plot(X4, delta_y)
        plt.plot(X5, delta_y2)
        plt.xlabel("Voltage")
        plt.ylabel("Delta Capacity")
        legend1 = "Cycle " + str(m) + " vs. " + str(n)
        legend2 = "Cycle " + str(m) + " vs. " + str(o)
        ax4.legend([legend1, legend2])
        plt.title(cell_ID)
        plt.savefig(output_path + cell_ID + '/' + cell_ID + "_DeltaQ(V)_cycle_10_100.png") 
        plt.show(block = False)
        plt.pause(0.5)
        plt.close()   
        plt.clf()  

    #print(cycle_df.head(50))
    #print(cycle_df.tail(50))
    #cycle_df.info()
    cycle_df.describe()
    
    #print(rawdata_df.head(50))
    #print(rawdata_df.tail(50))
    #rawdata_df.info()
    rawdata_df.describe()

end = time.time()
print(str((end - start)/60) + " minutes")

# features for the cell

# Cell_df
# 1. Cell ID
# 2. Chemistry: V_mean, V_std, V_IQR, V_max, V_min
# 3. Nominal Capacity: dis_Q_nom
# 4. C-rate: dis_C-rate_mode, dis_C-rate_max
# 5. cycle: cycles_tested, cycle_life (for 80% end-of-life)
# 6. Q(i) - i is cycle number (10, 25, 100)
#    a. cycle_i
#    b. Q_i (capacity at cycle i)
#    c. Q_deg_i (slope of capacity degration at cycle i - δQ/ δi)
# 7. ΔQ(i1=10 vs. i2); i2 = 25, 100
#    a. delta-Qi2_mean, delta-Qi2_std, delta-Qi2_IQR, delta-Qi2_max, delta-Qi2_min
# 8. ΔE(i1=10 vs. i2); i2 = 25, 100
#    a. delta-Ei2   
# 9. Columbic efficiency: Q_Eff_i
# 10. Energy efficiency: E_Eff_i
# 11. Capacity throughput: cumm_Q_i
# 12. Energy throughput: cumm_E_i
# 
#
# 13. Resistance
#    a. DCIR (i1=2, i2=100)
#    b. Current Interrupt - Discharge (On/Off), Charge (On/Off) (i1=2, i2=100)
#    c. ΔQ(V, i1=2, i2=100)
#    d. slope of resistance increase: δR/ δi (i1=2, i2=100)
# 

# 14. Temp (max), ∫(avg. temp)
# 15. Charging time
