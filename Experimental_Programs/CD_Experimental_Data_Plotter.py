"""
PHY420W
Single Chua's Diode Experimental Data Plots
Created by Jeryl Schudel
02/05/26

This program plots data from a Digilent WaveForms produced csv file to 
create a V-I characteristic curve for the Chua Diode.  The curve is fitted to 
the piecewise function defining the non-linear resistance.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

data = pd.read_csv("CD_040826_v4.csv") #Change this line to get different csv data files
time = np.array(data['Time (s)'].tolist())
V_sample = np.array(data['Channel 1 (V)'].tolist()) #Voltage over entire circuit
V_current = np.array(data['Channel 2 (V)'].tolist()) #Voltage over sampling resistor

time_label = '1s'
V_label = 'Voltage [V]'
I_label = 'Current [A]'

def piecewise(v, k1, k2, b1, b2):
    return ((k1*v) + (1/2)*(k2-k1)*(abs(v-b1)-abs(v-b2)))

initial = [1,1,1.3, -1.2]  #Initial guesses for the curve fitting
testresist = 100   #Circuit test resistance to calculate the current

V = (V_sample - V_current)
I = (V_current/testresist)

params, covariance = curve_fit(piecewise, V, I, p0=initial)

plt.plot(V, I, marker="o")
x_fine = np.linspace(-7, 7, 1000)
plt.plot(x_fine, piecewise(x_fine, *params), 'r-', label='Fitted curve')
plt.title('Chua\'s Diode Characteristic Curve (V-I) 12V')
plt.xlabel(V_label)
plt.ylabel(I_label)
plt.grid(True)
plt.show()

inner, outer, highbp, lowbp = params
print ("Inner piece slope: %10.2E" %(inner))
print ("Outer piece slope: %10.2E" %(outer))
print (f"High Breakpoint: {highbp:.2f}")
print (f"Low Breakpoint: {lowbp:.2f}")