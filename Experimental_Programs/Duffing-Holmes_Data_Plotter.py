"""
Duffing-Holmes circuit Data Plotting Program
Created by Jeryl Schudel
05Feb2026

This program is meant to plot data obtained from a Digilent WaveForms produced csv 
on the Duffing Holmes circuit.  The output is a phase space diagram.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


setdata = "450mV" #Change this line to get different csv data files
data = pd.read_csv("DH_"+setdata+"_Data.csv")
time = np.array(data['Time (s)'].tolist())
V_c = np.array(data['Channel 1 (V)'].tolist())


time_label = 'ms'
V_cap_label = 'Voltage [V]'
I_ind_label = 'Amperes'
dVdt_label = 'dV/dt [V/s]'

w_0 = 1.3

time = time*1000
time_sc = time/w_0


dV = [0]
dt = [0]

for i in range(len(V_c)-1):
    dV.append(V_c[i+1]-V_c[i])
    dt.append(time[i+1]-time[i])

dV = np.array(dV)
dt = np.array(dt)

dVdt = dV/dt


plt.plot(V_c, dVdt)
plt.title('a = '+ setdata +', b = 0.10, w = 1.3 rad/s')
plt.xlabel(V_cap_label)
plt.ylabel(dVdt_label)
plt.grid()
plt.show()
