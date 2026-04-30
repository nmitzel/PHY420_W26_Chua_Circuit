'''
Chua_Circuit_Evolution.py

Iterates over an array of values of dimensionless Chua circuit parameter alpha,
computing the numerical solution to the system of differential equations that govern 
a Chua circuit and extracting the local maximum values in its evolution (x_max).

Differential equations of motion & formulae for dimensionless parameters taken from
    
Liu et al. "Synchronization control between two Chua's circuits via capacitive coupling"
    Applied Mathematics and Computing 360 (2019) pp. 96-104.

Kennedy, "Robust Op-Amp realization of Chua's circuit" Frequenz vol 40 no. 3-4 
    (March-April 1992) pp. 66-80.

Adapted from "Duffing-Holmes_Bifurcation_Diagram_v0.py," written by Dr. Ernest Behringer 
(Dept. of Physics and Astronomy, Eastern Michigan University),
by the PHY420 class of Winter 2026.

'''

import numpy as np
from numpy import array,arange,vstack,transpose
#from numpy import savetxt
#from numpy import polyfit,polyder,polyval,roots
from numpy import zeros,remainder,sin
from pylab import plot,xlim,ylim,xlabel,ylabel,grid,show,figure,title,scatter,xticks,yticks
from scipy.integrate import odeint
np.seterr('warn')

# Specify input parameters
## The numbering of the resistors is taken from Kennedy (1992).

## Components of the conventional linear oscillation circuit
#C_1 = 10.0*(10**(-9)) #F
C_2 = 100.0*(10**(-9)) #F
R = 1.8*(10**3) #Ohms 
L = 18.0*(10**(-3)) #H
R_L = 0 #Ohms ## resistance associated with the inductor
## Resistors in the Chua diode
## first non-linear resistor
R_1 = 220.0 #Ohms
R_2 = R_1
R_3 = 2.2*(10**3) #Ohms
## second non-linear resistor
R_4 = 20.0*(10**3) #Ohms
R_5 = R_4
R_6 = 3.3*(10**3) #Ohms

# Dimensionless parameters
## The formulae for m_0 and m_1 are taken from Kennedy (1992) 
## with a factor of R included to make the values dimensionless.
## The formulae for beta and gamma are taken from Liu et al. (2019)

## first non-linear resistor
m_01 = 1.0/R_1
m_11 = -1.0/R_3
## second non-linear resistor
m_02 = 1.0/R_4
m_12 = -1.0/R_6
## full circuit
m_0 = (m_11 + m_12)*R
m_1 = (m_11 + m_02)*R
#print(m_01,m_11,m_02,m_12,m_0,m_1)

## The formulae for alpha, beta, and gamma are taken from equation 4 in Liu et al. (2019)
beta = (C_2*(R**2))/L
gamma = (R_L*R*C_2)/L
#print(beta,gamma)

## generate list of alpha values to test
alpha_min = 3.0 # minimum ratio of C2 to C1
alpha_max = 12.0 # maximum ratio of C2 to C1
alpha_step = 0.05 # step size for alpha #for best results (and a crazy long running time), use 0.01
nmax = 200 # maximum number of maxima to store
alpha_values = arange(alpha_min,alpha_max,alpha_step)
#print(alpha_values)

# Initialize lists that can be filled with values 
# This is the array of index values 
indicesOfMax=zeros((len(alpha_values),nmax))
# This is the list of good roots
goodRoots=[]
## This is the list of alpha values to plot
alpha_valuesplot=[]
# This is the list of corresponding position maxima to plot
Vmaxvalues=[]

# Initialize time interval endpoints and the number of time steps
# Calculate step size, then generate list of times at which  
# the voltage (position) and its time derivatives are reported
t1 = 0.0 # initial scaled time
t2 = 1600.0 # final scaled time
N_per_t_sc = 500 # previously 100 # number of time steps per scaled time interval
N = N_per_t_sc*(t2-t1) # number of time steps
h = (t2-t1)/N # time step size
tpoints = np.linspace(t1,t2,200000) # set of scaled times to report voltages
    

## The non-linear function f(x) and the differential equations
## As shown in Liu et al. (2019).
## Other literature may use different sign conventions.

def f_E(x):
    f_x = (m_1*x) + 0.5*(m_0 - m_1)*(abs(x+1)-abs(x-1))
    return f_x

def derivatives(r,t):
    x = r[0]
    y = r[1]
    z = r[2]
    
    # Prevents explosion
    if abs(x) > 100:
        x = np.sign(x)*10
    
    dxdt = alpha*(y - x) - alpha*f_E(x)
    dydt = x - y + z
    dzdt = -1*beta*y - gamma*z
    return array([dxdt, dydt, dzdt], float)

# Loop over alpha values
for j in range (0, len(alpha_values)-1):

    # Specify alpha
    alpha = alpha_values[j]

    # Specify initial conditions
    x0 = 0.1 ## dimensionless
    y0 = 0.1 ## dimensionless
    z0 = 0.01 ## dimensionless
    r0 = array([x0,y0,z0],np.float64)
    
    # Calculate the evolution of the voltage using SciPy odeint
    r,info = odeint(derivatives,r0,tpoints,mxstep=50000,full_output=True)
    print(info)
    #print(r)
    # Update the coefficients for the next loop cycle
    if j < len(alpha_values):
        alpha = alpha_values[j+1]

    #print len(r[:,0])
        
    #Output the data in columns
    #output_array_rows = vstack([tpoints,transpose(r[:,0]),transpose(r[:,1])])
    #savetxt("R_V_at_"+str(j)+".txt",output_array_rows)
    #output_array_columns = transpose(output_array_rows)
    #savetxt("R_V_at_"+str(j)+"_columns.txt",output_array_columns)
    
    # Now find the maxima that appear in the voltage evolution
    
    # initialize a counter for the number of maxima found for this evolution; 
    # it must be reset each time a new evolution has been calculated
    maxctr = 0;
    
    # Assign each row with time or V values
    # We only look at the latter one-eighth of the calculated evolution 
    if remainder(len(tpoints),2) > 0:
        # the number of points is odd, so
        timeData=tpoints[0:len(tpoints)] #Instead of 0 is used to be int(7*(len(tpoints)-1)/8)
        VData=transpose(r[0:len(tpoints),0])
    else:
        # the number of points is even, so
        timeData=tpoints[0:len(tpoints)]
        VData=transpose(r[0:len(tpoints),0])
    #print timeData
    
    # Find the number of data points, to set the number of loop iterations
    numTimes=timeData.size
    #print numTimes
    
    # Create an array of time and V data generated by odeint
    outputvTime=vstack([timeData,VData])
    
    # Declare initial values for loop to find data maxima
    initialTimeIndex=0
    initialVIndex=0
    
    # Initialize the values of the array indices for three consecutive points
    timeIndex1=initialTimeIndex
    VIndex1=initialVIndex
    timeIndex2=initialTimeIndex+1
    VIndex2=initialVIndex+1
    timeIndex3=initialTimeIndex+2
    VIndex3=initialVIndex+2
    
    # FIND THE MAXIMA IN THE AMPLITUDE VS TIME DATA  
    # by using three consecutive points (a "trio" of points) 
    # to calculate the slope of the corresponding line segments.
    # Then test whether the respective slopes 
    # are positive and then negative.
    # If true, save the index of the second of the three points,
    # and save the a value and the voltage (position) value at the index, 
    # and increment the maximum counter (maxctr).
    # Loop over all the (time,voltage) pairs in the evolution
    for i in range(0,numTimes-4):
        # check a trio of points
        deltaV1 = VData[VIndex2]-VData[VIndex1]
        deltat1 = timeData[timeIndex2]-timeData[timeIndex1]
        slope1 = deltaV1/deltat1
        deltaV2 = VData[VIndex3]-VData[VIndex2]
        deltat2 = timeData[timeIndex3]-timeData[timeIndex2]
        slope2 = deltaV2/deltat2
    
        # Test whether the slope values go from pos to neg, indicating max
        if slope1>0 and slope2<0:
            # save the index of the time array element
            indicesOfMax[j]=timeIndex2 # this line was originally indicesOfMax[j][maxctr]=timeIndex2. What was maxctr doing there?
            # save the value of a that produced the maximum
            alpha_valuesplot.append(alpha)
            # save the value of the maximum
            Vmaxvalues.append(VData[timeIndex2])
            # increment the maximum counter
            maxctr+=1  
            #print indicesOfMax
    
        # Increment the indices after the test
        # to check the next trio of points
        timeIndex1+=1
        VIndex1+=1
        timeIndex2+=1
        VIndex2+=1
        timeIndex3+=1
        VIndex3+=1
        
# Plot the bifurcation diagram
figure("Single Chua Circuit Bifurcation Diagram",figsize=(6,5))
alpha_valuesplot2 = array(alpha_valuesplot,float)
scatter(alpha_valuesplot2,Vmaxvalues,color="cyan",s=0.01)
xlim(alpha_min,alpha_max)
#ylim()
xlabel(r"$\alpha$",fontsize=20)
ylabel(r"$x_{max}$",fontsize=20)
xticks(fontsize=18)
yticks(fontsize=18)
title(r'$ \beta =$ %.3f, $m_0 =$ %.3f, $m_1 = $ %.3f'%(beta,m_0,m_1),fontsize=14)
grid(True)
show()
